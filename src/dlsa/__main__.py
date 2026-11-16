"""Run one stage or the complete fixed experiment."""
import argparse
import os
from pathlib import Path

from .config import ARCHITECTURES, DATA_DIR, RESULTS_DIR, RUN_DIR, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "backtest", "plot", "reproduce"):
        command = commands.add_parser(name)
        if name == "plot":
            command.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
            command.add_argument("--output-dir", type=Path, default=RUN_DIR / "report")
        else:
            command.add_argument("--run-dir", type=Path, default=RUN_DIR)
        if name in ("prepare", "reproduce"):
            command.add_argument("--data-dir", type=Path, default=DATA_DIR)
        if name in ("train", "backtest"):
            command.add_argument("--architecture", choices=("all", *ARCHITECTURES), default="all")
            command.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
        if name in ("train", "backtest", "reproduce"):
            command.add_argument("--device", default="auto", help="auto uses CPU Fourier and CUDA CNN")
        if name == "backtest":
            command.add_argument("--checkpoints", type=Path, help="default: RUN_DIR/checkpoints")
    args = parser.parse_args()

    # Set before importing NumPy/PyTorch or initializing their thread pools.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"

    if args.command == "plot":
        from .plots import make_report
        make_report(args.results_dir, args.output_dir)
        return
    if args.command in ("prepare", "reproduce"):
        from .data import prepare
        prepare(args.data_dir, args.run_dir)
    if args.command in ("train", "backtest"):
        architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
        seeds = tuple(args.seeds)
    else:
        architectures, seeds = ARCHITECTURES, SEEDS
    if args.command in ("train", "reproduce"):
        from .train import train
        train(args.run_dir, architectures, seeds, args.device)
    if args.command in ("backtest", "reproduce"):
        from .backtest import backtest
        checkpoints = args.checkpoints if args.command == "backtest" else None
        backtest(args.run_dir, checkpoints, architectures, seeds, args.device)
    if args.command == "reproduce":
        from .plots import make_report
        make_report(args.run_dir, args.run_dir / "report")


if __name__ == "__main__":
    main()
