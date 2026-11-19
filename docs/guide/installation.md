# Installation

reguq needs Python ≥ 3.10.

```bash
pip install "git+https://github.com/DaneshSelwal/reguq"          # core
pip install "reguq[boosting,tuning] @ git+https://github.com/DaneshSelwal/reguq"
```

Once released on PyPI, use `pip install reguq` and `pip install "reguq[boosting,tuning]"`.

## Optional extras

The core install has four dependencies (numpy, scipy, pandas, scikit-learn). Everything else is
optional. If an estimator needs a missing backend, it raises an `ImportError` that names the extra to
install.

| Extra | Installs | Needed for |
|---|---|---|
| `boosting` | lightgbm, xgboost, catboost | boosted base models and quantile objectives |
| `gpboost` | gpboost | GPBoost base model |
| `ngboost` | ngboost | `NGBoostRegressor` |
| `pgbm` | pgbm, torch | `PGBMRegressor` |
| `torch` | torch | `CARDRegressor`, `HCMRegressor`, ACP with `policy="mlp"` |
| `tabnet` | pytorch-tabnet, torch | `TabNetRegressor` |
| `qrf` | quantile-forest | `QuantileForestRegressor` |
| `treeffuser` | treeffuser | `TreeffuserRegressor` |
| `tuning` | optuna, optunahub | `reguq.tuning`, pipeline tuning |
| `report` | matplotlib, openpyxl, XlsxWriter, PyYAML | `reguq.reporting`, YAML configs |
| `mapie`, `puncc` | mapie, puncc | cross-checking against other libraries |
| `explain` | shap, lime | model explanations |
| `all` | everything above | |
| `docs`, `dev` | MkDocs; pytest, ruff, mypy, pre-commit | contributing |

!!! tip "CPU-only torch"
    For a smaller install, install the CPU torch wheel before the extras:
    `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Verifying the install

```bash
python -c "import reguq, sys; print(reguq.__version__); assert 'torch' not in sys.modules"
```
