# Recorded experiment

These files preserve the completed experiment: 1,488 fits, 2,976 saved model
states, and 96 deployed return paths. They were exported from the original
files without training or model inference. `provenance.json` records the
export checks and hashes; `source_manifest.json.gz` maps normalized source
identifiers to original file hashes. These identifiers are provenance labels,
not literal filesystem paths or runtime dependencies.

All NPZ files load with `numpy.load(path, allow_pickle=False)`. Their named axes
specify ordering explicitly; all arrays retain their original numeric precision.

## Returns and histories

`daily.npz` contains `gross`, `net`, `turnover`, and `shorts`, each with axes
`(seed, architecture, arm, policy, date)` and shape `(3, 2, 4, 4, 3772)`.
The matching axis arrays are `seed`, `architecture`, `arm`, `policy`, and
`dates`. Gross/net returns are daily excess returns in decimal units;
turnover is the stock-level L1 change, and shorts is total short exposure.

`history.npz` contains `values` with axes
`(seed, architecture, arm, branch, block, epoch, split, metric)`.
There are 31 refits, 100 epochs, and train/validation splits. `branch` is
`full` or `rolling`; full-branch validation is absent and stored as NaN.
The `metric` array names all 13 scalar fields, including Sharpe, return
moments, turnover, optimizer statistics, holdings coefficients, and timing.
Fields absent in an original history are NaN. `batch_seconds` adds an
eight-element minibatch axis to the first seven axes; short windows use
six entries, and unused or historically absent entries are NaN.
Together these arrays preserve all 223,200 original history rows.

`metrics.csv` retains the original per-seed statistics for every architecture,
arm, policy, and gross/net basis. Each Mean row is the arithmetic average of
the three seed values for SR, CAGR, maximum drawdown, and ending index.
Drawdown is a positive peak-to-trough loss computed separately on each full
seed path before averaging. The plotted Mean wealth curve is the pointwise
average of seed indices; its CAGR and drawdown are not used in the Mean row.
Session and audit-block counts remain metadata. The aggregation change is
recorded in `provenance.json`; daily returns and fitted models are unchanged.

## Model states and selections

Each `checkpoints/{architecture}_seed{seed}.npz` contains 496 original saved
model states. Metadata arrays are `architecture`, `seed`, `arm`, `branch`,
`block`, `checkpoint`, `epochs`, and `source_sha256`.

- `branch`: `full`, `rolling`.
- `checkpoint`: `selected`, `epoch100`.
- `epochs` and `source_sha256`: shape `(4, 2, 31, 2)`, indexed by arm,
  branch, block, and checkpoint.
- Each parameter key is `{arm}/{original_state_dict_parameter_name}`.
  Its array has shape `(2, 31, 2, *parameter_shape)`, indexed by branch,
  block, checkpoint, then the original parameter dimensions.

The policy mapping is:

| Policy | Branch | Checkpoint |
| --- | --- | --- |
| `full_epoch100` | `full` | `epoch100` |
| `initial_selected` | `full` | `selected` |
| `rolling_epoch100` | `rolling` | `epoch100` |
| `rolling_selected` | `rolling` | `selected` |

Parameter names, dtype, shape, tensor values, and selected epochs are retained.
Optimizer state and runtime recovery files are omitted. To inspect a state:

```python
import numpy as np

saved = np.load("results/checkpoints/cnn_seed0.npz", allow_pickle=False)
arm = "current_recurrent"
state = {
    key.split("/", 1)[1]: saved[key][1, 0, 0]
    for key in saved.files if key.startswith(arm + "/")
}
# state is the first rolling refit's selected model, as NumPy arrays.
```

`selections.csv` records every fit's epoch selection and checkpoint hashes.
`deployment_selections.csv` links all policy/refit deployments to their saved
checkpoint. `initial_selections.json` retains the initial fixed epochs.
`fit_metadata.json.gz` retains each original fit manifest and selection record,
including runtime and experiment settings; source-directory labels use the
same normalized identifiers as the source manifest.

`audit/` preserves the historical CNN audit summaries and the three recorded
FP32 discrepancies. Their strict accounting and recurrence checks passed;
none of the recorded paths was excluded or changed during packaging.
