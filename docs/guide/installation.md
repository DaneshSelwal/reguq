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
| `pgbm` | pgbm, torch | `PGBMRegressor` (**Python ≤ 3.11**: pgbm 2.3 does not support newer Python) |
| `torch` | torch | `CARDRegressor`, `HCMRegressor`, ACP with `policy="mlp"` |
| `tabnet` | pytorch-tabnet, torch | `TabNetRegressor` |
| `qrf` | quantile-forest | `QuantileForestRegressor` (install a quantile-forest release built for your scikit-learn; 1.5 needs scikit-learn ≥ 1.9) |
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

## Offline environments (e.g. Kaggle without internet)

Build the wheel with `pip wheel --no-deps .`, then download the backend wheels for the target platform:

```bash
pip download --no-deps --only-binary=:all: --platform manylinux_2_28_x86_64 \
    --python-version 3.13 --implementation cp -d wheels ngboost lifelines quantile-forest gpboost
```

Upload them as a dataset and install with `pip install --no-index --find-links wheels --no-deps …`.
