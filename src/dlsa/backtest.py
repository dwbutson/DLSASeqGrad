"""Causal stock holdings and daily excess returns, before and after fees."""
from pathlib import Path
import shutil

import numpy as np
import torch

from .config import (
    ARCHITECTURES, ARMS, BRANCHES, CHECKPOINTS, EXPERIMENT, POLICIES, RUN_DIR, SEEDS,
)
from .data import Cache
from .models import Policy, batches, configure, stock_weights


def accounting(weights, excess, previous=None, cost=None, borrow=None):
    cost = EXPERIMENT.turnover_cost if cost is None else cost
    borrow = EXPERIMENT.short_cost if borrow is None else borrow
    previous = np.zeros(weights.shape[1]) if previous is None else previous
    turnover = np.abs(np.diff(np.vstack([previous, weights]), axis=0)).sum(axis=1)
    shorts = np.maximum(-weights, 0).sum(axis=1)
    gross = (weights * excess).sum(axis=1)
    return {
        "gross": gross,
        "net": gross - cost * turnover - borrow * shorts,
        "turnover": turnover,
        "shorts": shorts,
    }


def return_metrics(returns):
    mean = float(np.mean(returns))
    deviation = float(np.std(returns, ddof=1))
    return {
        "daily_mean": mean,
        "daily_std": deviation,
        "sharpe": np.sqrt(252) * mean / deviation,
    }


@torch.no_grad()
def predict(model, data, previous=None):
    """Batch the encoder; recur only through the inexpensive allocation head."""
    model.eval()
    if model.architecture == "fourier" and not model.holdings:
        return np.concatenate([
            stock_weights(model(data.inputs[part]), data.B[part], data.C[part]).cpu().numpy()
            for part in batches(len(data))
        ])
    pieces = []
    for part in batches(len(data)):
        signals = model.encode(data.inputs[part])
        if model.holdings:
            value = torch.nn.functional.linear(
                signals, model.head.weight[:, :-1], model.head.bias
            ).squeeze(-1)
        else:
            value = model.allocate(signals)
        pieces.append(value.cpu().numpy())
    base = np.concatenate(pieces)
    coefficient = np.float32(model.head.weight[0, -1].item()) if model.holdings else np.float32(0)
    prior = np.zeros(base.shape[1]) if previous is None else previous.copy()
    weights = np.empty(base.shape, dtype=np.float64)
    for day in range(len(base)):
        # Match the fitted model: holding input and linear head are FP32;
        # projection, normalization, and fee accounting are FP64.
        q = base[day] + coefficient * prior.astype(np.float32)
        q64 = q.astype(np.float64)
        raw = q64 - (q64 @ data.b_cpu[day]) @ data.c_cpu[day].T
        gross = np.abs(raw).sum()
        weights[day] = raw / (gross if gross else 1.0)
        prior = weights[day]
    return weights


def load_model(archive, architecture, arm, branch, block, checkpoint, seed, device):
    """Load either an exported original fit or a fit from the compact trainer."""
    model = Policy(architecture, holdings=arm != "signal_only", seed=seed).to(device)
    index = (BRANCHES.index(branch), block, CHECKPOINTS.index(checkpoint))
    state = {
        name: torch.from_numpy(archive[f"{arm}/{name}"][index].copy())
        for name in model.state_dict()
    }
    model.load_state_dict(state)
    return model


def backtest(run_dir=RUN_DIR, checkpoints=None, architectures=ARCHITECTURES, seeds=SEEDS, device="auto"):
    run_dir = Path(run_dir)
    checkpoints = run_dir / "checkpoints" if checkpoints is None else Path(checkpoints)
    dates = np.load(run_dir / "prepared" / "dates.npy")
    from .data import read_splits
    splits = read_splits(run_dir / "prepared" / "splits.csv")
    first, last = splits[0]["test_start"], splits[-1]["test_stop"]
    shape = (len(SEEDS), len(ARCHITECTURES), len(ARMS), len(POLICIES), last - first)
    destination = run_dir / "daily.npz"
    fields = ("gross", "net", "turnover", "shorts")
    if destination.exists():
        with np.load(destination) as prior:
            output = {name: prior[name].copy() for name in fields}
    else:
        output = {name: np.full(shape, np.nan) for name in fields}
    for architecture in architectures:
        compute_device = configure(architecture, device, arm="previous_epoch")
        cache = Cache(run_dir, architecture, compute_device)
        for seed in seeds:
            with np.load(checkpoints / f"{architecture}_seed{seed}.npz") as archive:
                for arm_index, arm in enumerate(ARMS):
                    for policy_index, (branch, checkpoint) in enumerate(POLICIES.values()):
                        previous = np.zeros(cache.arrays["x"].shape[1])
                        for split in splits:
                            model = load_model(
                                archive, architecture, arm, branch, split["block"],
                                checkpoint, seed, compute_device,
                            )
                            data = cache.dataset(split["test_start"], split["test_stop"])
                            weights = predict(model, data, previous)
                            daily = accounting(weights, data.x_cpu, previous)
                            index = (
                                SEEDS.index(seed), ARCHITECTURES.index(architecture), arm_index,
                                policy_index, slice(split["test_start"] - first, split["test_stop"] - first),
                            )
                            for name in fields:
                                output[name][index] = daily[name]
                            # Test holdings carry across refits, even when the model changes.
                            previous = weights[-1].copy()
            print(f"Backtested {architecture}, seed {seed}", flush=True)
        del cache
    np.savez_compressed(
        destination, **output, seed=SEEDS, architecture=ARCHITECTURES,
        arm=ARMS, policy=tuple(POLICIES), dates=dates[first:last],
    )
    source_history = checkpoints.parent / "history.npz"
    target_history = run_dir / "history.npz"
    if source_history.resolve() != target_history.resolve():
        shutil.copyfile(source_history, target_history)
