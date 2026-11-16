"""Four learning rules, two fitting windows, and two retained checkpoints."""
import json
from pathlib import Path
import random
import time
from typing import List, Tuple

import numpy as np
import torch

from .backtest import accounting, predict, return_metrics
from .config import (
    ARCHITECTURES, ARMS, BRANCHES, CHECKPOINTS, EXPERIMENT, HISTORY_METRICS, RUN_DIR, SEEDS,
)
from .data import Cache
from .models import Policy, batches, configure, stock_weights


def sequence(
    signals: torch.Tensor, b: torch.Tensor, c: torch.Tensor, x: torch.Tensor,
    previous: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
    cost: float, borrow: float, detached: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Current holdings with stopped or full within-minibatch gradients."""
    steps = torch.unbind(signals, dim=0)
    targets = torch.jit.annotate(List[torch.Tensor], [])
    rewards = torch.jit.annotate(List[torch.Tensor], [])
    previous = previous.detach()
    for day in range(signals.size(0)):
        if detached:
            previous = previous.detach()
        inputs = torch.cat((steps[day], previous.to(signals.dtype).unsqueeze(-1)), dim=-1)
        q = torch.nn.functional.linear(inputs, weight, bias).squeeze(-1).to(b.dtype)
        raw = q - (q @ b[day]) @ c[day].t()
        gross = raw.abs().sum()
        target = raw / torch.where(gross == 0.0, torch.ones_like(gross), gross)
        reward = (
            (target * x[day]).sum()
            - cost * (target - previous).abs().sum()
            - borrow * (-target).clamp_min(0.0).sum()
        )
        rewards.append(reward)
        targets.append(target)
        previous = target
    return torch.stack(rewards), torch.stack(targets)


def train_epoch(model, optimizer, data, arm, frozen, sequence_fn):
    cfg = EXPERIMENT
    model.train()
    cost, borrow = (0.0, 0.0) if arm == "signal_only" else (cfg.turnover_cost, cfg.short_cost)
    prior = torch.zeros(data.x.shape[1], dtype=data.B.dtype, device=data.x.device)
    if arm == "previous_epoch":
        lag = np.vstack([np.zeros(data.x.shape[1]), frozen[:-1]])
        lagged = torch.from_numpy(lag).to(data.x.device)
    losses, returns, gradient_norms = [], [], []
    floor_batches = 0
    for part in batches(len(data)):
        if arm == "previous_epoch" or (arm == "signal_only" and model.architecture == "cnn"):
            previous = lagged[part] if arm == "previous_epoch" else torch.zeros_like(data.x[part])
            signals = model.encode(data.inputs[part])
            q = model.allocate(signals, previous if model.holdings else None)
            weights = stock_weights(q, data.B[part], data.C[part], batched=True)
            pnl = (
                (weights * data.x[part]).sum(dim=1)
                - cost * (weights - previous).abs().sum(dim=1)
                - borrow * (-weights).clamp_min(0).sum(dim=1)
            )
        elif arm == "signal_only":
            weights = stock_weights(model(data.inputs[part]), data.B[part], data.C[part])
            pnl = (weights * data.x[part]).sum(dim=-1)
        else:
            pnl, weights = sequence_fn(
                model.encode(data.inputs[part]), data.B[part], data.C[part], data.x[part],
                prior, model.head.weight, model.head.bias, cost, borrow, arm == "current_detached",
            )
        deviation = pnl.std(correction=1)
        loss = -pnl.mean() / deviation.clamp_min(cfg.volatility_floor)
        floor_batches += int(deviation.detach() < cfg.volatility_floor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norms.append(torch.sqrt(sum(
            (parameter.grad.detach().double() ** 2).sum() for parameter in model.parameters()
        )))
        losses.append(loss.detach())
        returns.append(pnl.detach())
        # Carry state values across optimizer updates, but truncate their graphs.
        prior = weights[-1].detach()
        optimizer.step()
    return {
        "gradient_norm_max": float(torch.stack(gradient_norms).max().cpu()),
        "loss_floor_batches": floor_batches,
        "surrogate_batch_loss_mean": float(torch.stack(losses).mean().cpu()),
        "surrogate_sharpe": return_metrics(torch.cat(returns).cpu().numpy())["sharpe"],
    }


def model_state(model):
    return {name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}


def fit(training, validation, architecture, arm, seed, sequence_fn, fixed_epoch=None):
    cfg = EXPERIMENT
    configure(architecture, str(training.x.device), arm)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if training.x.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Policy(architecture, arm != "signal_only", seed).to(training.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    frozen = np.zeros(training.x.shape) if arm == "previous_epoch" else None
    history = np.full((cfg.epochs, 2, len(HISTORY_METRICS)), np.nan)
    best_score, selected_epoch, selected = -np.inf, fixed_epoch, None
    train_fees = (0.0, 0.0) if arm == "signal_only" else (cfg.turnover_cost, cfg.short_cost)
    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        statistics = train_epoch(model, optimizer, training, arm, frozen, sequence_fn)
        datasets = [training] if validation is None else [training, validation]
        for split_index, data in enumerate(datasets):
            weights = predict(model, data)
            if split_index == 0 and arm == "previous_epoch":
                # Refresh in dropout-off mode; freeze the entire trajectory next epoch.
                frozen = weights.copy()
            fees = train_fees if split_index == 0 else (cfg.turnover_cost, cfg.short_cost)
            daily = accounting(weights, data.x_cpu, cost=fees[0], borrow=fees[1])
            row = {
                **return_metrics(daily["net"]), **statistics,
                "gross_sharpe": return_metrics(daily["gross"])["sharpe"],
                "turnover": daily["turnover"].mean(),
                "short_exposure": daily["shorts"].mean(),
                "optimizer_updates": epoch * len(list(batches(len(training)))),
                "epoch_compute_seconds": time.perf_counter() - started,
                "holdings_coefficient": model.head.weight[0, -1].item() if model.holdings else 0.0,
            }
            history[epoch - 1, split_index] = [row[name] for name in HISTORY_METRICS]
        if validation is not None:
            score = row["sharpe"]
            # Strict improvement retains the first epoch in an exact tie.
            if score > best_score:
                best_score, selected_epoch = score, epoch
                selected = model_state(model)
        elif epoch == fixed_epoch:
            selected = model_state(model)
    return {"selected": selected, "epoch100": model_state(model)}, selected_epoch, history


def save_history(path, values, blocks):
    np.savez_compressed(
        path, values=values, seed=SEEDS, architecture=ARCHITECTURES, arm=ARMS,
        branch=BRANCHES, block=np.arange(blocks), epoch=np.arange(1, EXPERIMENT.epochs + 1),
        split=("train", "validation"), metric=HISTORY_METRICS,
    )


def train(run_dir=RUN_DIR, architectures=ARCHITECTURES, seeds=SEEDS, device="auto"):
    """Train all requested fits; completed architecture/seed archives are reused."""
    run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sequence_fn = torch.jit.script(sequence)
    history_path = run_dir / "history.npz"
    histories = None
    if history_path.exists():
        with np.load(history_path) as saved:
            histories = saved["values"].copy()
    runtime_path = run_dir / "runtime.json"
    runtimes = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    for architecture in architectures:
        compute_device = configure(architecture, device)
        cache = Cache(run_dir, architecture, compute_device)
        blocks = len(cache.splits)
        if histories is None:
            histories = np.full(
                (len(SEEDS), len(ARCHITECTURES), len(ARMS), len(BRANCHES), blocks,
                 EXPERIMENT.epochs, 2, len(HISTORY_METRICS)), np.nan,
            )
        runtimes[architecture] = {
            "numpy": np.__version__, "torch": torch.__version__, "device": str(compute_device),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(compute_device) if compute_device.type == "cuda" else None,
        }
        for seed in seeds:
            archive_path = checkpoint_dir / f"{architecture}_seed{seed}.npz"
            if archive_path.exists():
                print(f"Using completed {architecture}, seed {seed}", flush=True)
                continue
            epochs = np.zeros((len(ARMS), len(BRANCHES), blocks, len(CHECKPOINTS)), dtype=np.int16)
            archive = {}
            for arm_index, arm in enumerate(ARMS):
                initial_epoch = None
                # The first rolling fit supplies the constant epoch for all full fits.
                for branch in ("rolling", "full"):
                    branch_index = BRANCHES.index(branch)
                    for split in cache.splits:
                        block = split["block"]
                        start, stop = split["fit_start"], split["fit_stop"]
                        if branch == "rolling":
                            boundary = start + EXPERIMENT.rolling_training_days
                            training = cache.dataset(start, boundary)
                            validation = cache.dataset(boundary, stop)
                        else:
                            training = cache.dataset(start, stop)
                            validation = None
                        states, selected_epoch, history = fit(
                            training, validation, architecture, arm, seed, sequence_fn,
                            fixed_epoch=initial_epoch,
                        )
                        if branch == "rolling" and block == 0:
                            initial_epoch = selected_epoch
                        index = (SEEDS.index(seed), ARCHITECTURES.index(architecture), arm_index, branch_index, block)
                        histories[index] = history
                        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                            epochs[arm_index, branch_index, block, checkpoint_index] = (
                                selected_epoch if checkpoint == "selected" else EXPERIMENT.epochs
                            )
                            for name, value in states[checkpoint].items():
                                key = f"{arm}/{name}"
                                if key not in archive:
                                    archive[key] = np.empty((len(BRANCHES), blocks, len(CHECKPOINTS), *value.shape), dtype=value.dtype)
                                archive[key][branch_index, block, checkpoint_index] = value
                        print(f"Fitted {architecture}, seed {seed}, {arm}, {branch}, block {block}", flush=True)
            save_history(history_path, histories, blocks)
            np.savez_compressed(
                archive_path, **archive, epochs=epochs, arm=ARMS, branch=BRANCHES,
                block=np.arange(blocks), checkpoint=CHECKPOINTS,
                seed=np.array(seed), architecture=np.array(architecture),
            )
            runtime_path.write_text(json.dumps(runtimes, indent=2) + "\n", encoding="utf-8")
        del cache
