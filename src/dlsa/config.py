"""The single experiment represented by the published figures."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUN_DIR = ROOT / "runs" / "reproduction"

SEEDS = (0, 1, 2)
ARCHITECTURES = ("fourier", "cnn")
ARMS = ("signal_only", "previous_epoch", "current_detached", "current_recurrent")
BRANCHES = ("full", "rolling")
CHECKPOINTS = ("selected", "epoch100")
POLICIES = {
    "full_epoch100": ("full", "epoch100"),
    "initial_selected": ("full", "selected"),
    "rolling_epoch100": ("rolling", "epoch100"),
    "rolling_selected": ("rolling", "selected"),
}
HISTORY_METRICS = (
    "daily_mean", "daily_std", "sharpe", "gross_sharpe", "turnover",
    "short_exposure", "optimizer_updates", "gradient_norm_max",
    "loss_floor_batches", "surrogate_batch_loss_mean", "surrogate_sharpe",
    "epoch_compute_seconds", "holdings_coefficient",
)


@dataclass(frozen=True)
class Experiment:
    factors: int = 5
    pca_window: int = 252
    loading_window: int = 60
    lookback: int = 30
    full_training_days: int = 970
    rolling_training_days: int = 720
    validation_days: int = 250
    refit_days: int = 125
    epochs: int = 100
    batch_size: int = 125
    learning_rate: float = 0.001
    dropout: float = 0.25
    volatility_floor: float = 1e-8
    turnover_cost: float = 0.0001
    short_cost: float = 0.00001


EXPERIMENT = Experiment()
