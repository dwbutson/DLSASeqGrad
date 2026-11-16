"""Past-only PCA, cumulative residual paths, and the prepared training cache."""
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
import torch

from .config import DATA_DIR, EXPERIMENT, RUN_DIR

INDEX_COLUMNS = (
    "envelope_start", "envelope_stop", "fit_start", "fit_stop", "test_start", "test_stop"
)


def read_splits(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return [
            {key: int(row[key]) for key in ("block", *INDEX_COLUMNS)}
            for row in csv.DictReader(stream)
        ]


def pca(history):
    """Five correlation factors, with loadings fitted on the last 60 days."""
    cfg = EXPERIMENT
    volatility = history.std(axis=0, ddof=0)
    standardized = (history - history.mean(axis=0)) / volatility
    eigenvalues, eigenvectors = eigh(
        standardized @ standardized.T / cfg.pca_window,
        subset_by_index=[cfg.pca_window - cfg.factors, cfg.pca_window - 1],
        driver="evr",
    )
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    directions = standardized.T @ eigenvectors / np.sqrt(cfg.pca_window * eigenvalues)
    factor_portfolios = directions / volatility[:, None]
    recent = history[-cfg.loading_window:]
    factor_returns = recent @ factor_portfolios
    loadings = np.linalg.lstsq(factor_returns, recent, rcond=None)[0].T
    return loadings, factor_portfolios


def prepare(data_dir=DATA_DIR, run_dir=RUN_DIR):
    """Rebuild derived arrays from the fixed raw-return panel; never redownload."""
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    prepared = run_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    cfg = EXPERIMENT
    with np.load(data_dir / "returns.npz") as panel:
        excess = panel["r"] - panel["rf"][:, None]
        dates = panel["dates"].copy()
    splits = read_splits(data_dir / "splits.csv")
    offset = min(row["envelope_start"] for row in splits)
    rows, stocks = excess[offset:].shape

    def array(name, shape, dtype):
        return np.lib.format.open_memmap(
            prepared / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )

    loadings = array("B", (rows, stocks, cfg.factors), np.float64)
    portfolios = array("C", (rows, stocks, cfg.factors), np.float64)
    residual = array("residual", (rows, stocks), np.float64)
    for row, date_index in enumerate(range(offset, len(excess))):
        b, c = pca(excess[date_index - cfg.pca_window:date_index])
        loadings[row], portfolios[row] = b, c
        residual[row] = excess[date_index] - b @ (c.T @ excess[date_index])
    paths = array("paths", (rows, stocks, cfg.lookback), np.float32)
    features = array("features", paths.shape, np.float32)
    paths[:cfg.lookback] = 0
    features[:cfg.lookback] = 0
    for row in range(cfg.lookback, rows):
        # Accumulate in FP64 before casting, as in the original feature cache.
        paths[row] = residual[row - cfg.lookback:row].cumsum(axis=0).T.astype(np.float32)
        fourier = np.fft.rfft(paths[row], axis=-1)
        features[row] = np.concatenate([fourier.real, fourier.imag[..., 1:-1]], axis=-1)
    for values in (loadings, portfolios, residual, paths, features):
        values.flush()
    np.save(prepared / "x.npy", excess[offset:])
    np.save(prepared / "dates.npy", dates[offset:])
    for row in splits:
        for column in INDEX_COLUMNS:
            row[column] -= offset
    with (prepared / "splits.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("block", *INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(splits)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "raw_panel_offset": offset}, indent=2) + "\n",
        encoding="utf-8",
    )


@dataclass
class Dataset:
    inputs: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    x: torch.Tensor
    b_cpu: np.ndarray
    c_cpu: np.ndarray
    x_cpu: np.ndarray

    def __len__(self):
        return len(self.x)


class Cache:
    """Keep the chosen encoder inputs and low-rank factors on the compute device."""

    def __init__(self, run_dir, architecture, device):
        self.directory = Path(run_dir) / "prepared"
        key = "paths" if architecture == "cnn" else "features"
        self.arrays = {
            name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
            for name in (key, "B", "C", "x", "dates")
        }
        self.inputs = torch.from_numpy(np.array(self.arrays[key], copy=True)).to(device)
        self.tensors = {
            name: torch.from_numpy(np.array(self.arrays[name], copy=True)).to(device)
            for name in ("B", "C", "x")
        }
        self.splits = read_splits(self.directory / "splits.csv")

    def dataset(self, start, stop):
        part = slice(start, stop)
        return Dataset(
            self.inputs[part], *(self.tensors[name][part] for name in ("B", "C", "x")),
            *(self.arrays[name][part] for name in ("B", "C", "x")),
        )
