# Pipeline and CLI

The pipeline runs every *model × method* combination on a train/test CSV pair.

1. Load the data and strip whitespace from column names.
2. Tune each model **on the training data only** (Optuna, CV or forward chaining).
3. Fit every method on the training data. Each method holds out its own calibration data.
4. Evaluate once on the test set with the same metric functions for every method.
5. Export the results.

## From Python

```python
from reguq.pipeline import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    train_path="train.csv", test_path="test.csv", target="St",
    time_ordered=True, alpha=0.1, calib_size=0.25,
    models=["lightgbm", "catboost", "randomforest"],
    methods=["split", "normalized", "cqr", "cv_plus", "nexcp", "aci", "dtaci"],
    model_free_methods=["ngboost", "qrf"],
    tuning={"n_trials": 40, "n_splits": 5, "sampler": "tpe", "pruner": "median"},
    output_dir="results", export_excel=True, export_plots=True,
)
res = run_pipeline(cfg)
res.metrics.sort_values("interval_score")
```

`PipelineConfig` validates itself when it is created. An unknown model or method, or an α outside
(0, 1), raises immediately, before any data is loaded.

## From the command line

```yaml
# config.yaml
train_path: data/train.csv
test_path: data/test.csv
time_ordered: true
alpha: 0.1
models: [lightgbm, xgboost]
methods: [split, cqr, aci]
tuning: {n_trials: 30, n_splits: 5}
output_dir: results
```

```bash
reguq-run config.yaml          # -v for debug logging
```

## Names

**Models** (`reguq.models.available_models()`): `linear`, `randomforest`, `extratrees`,
`gradientboosting`, `histgradientboosting`, `lightgbm`, `xgboost`, `catboost`, `gpboost`, `tabnet`.
Register your own with `reguq.models.register_model`.

**Methods that wrap a model** (`pipeline.METHODS`): `split`, `normalized`, `cqr`, `cv_plus`, `jab`,
`nexcp`, `aci`, `dtaci`, `sf_ogd`, `saocp`, `nexcp_online`, `extreme`, `acp`, `multi_quantile`,
`ibug`, `card`.

**Model-free methods** (`pipeline.MODEL_FREE`): `ngboost`, `pgbm`, `hcm`, `treeffuser`, `qrf`.

Some combinations do not exist. For example, CQR needs a model with a quantile objective, which
random forest does not have, and IBUG needs a tree ensemble whose leaves can be read. Such rows are
reported with `status = "error: …"` and are never replaced by a different method.

## Outputs

| File | Content |
|---|---|
| `metrics.csv` | one row per model × method: point, interval and distributional metrics, status, run time |
| `predictions/<method>__<model>.csv` | `y_true, y_pred, lower, upper` per test point |
| `params.json` | tuned hyper-parameters and the tuning log (best CV RMSE, trials, pruned trials) |
| `config.json` | the exact configuration, for reproducibility |
| `results.xlsx` | (with `export_excel`) metrics sheet + one sheet per prediction table |
| `plots/*.png` | (with `export_plots`) coverage-vs-width overview and one interval plot per run |
