# reguq — regression uncertainty quantification with tested guarantees

`reguq` turns any scikit-learn style regressor into one that outputs **prediction intervals** or
full **predictive distributions**. It implements conformal prediction, quantile regression and
probabilistic regression as the original papers define them, and checks the guarantees they claim
with Monte-Carlo tests.

> **Status:** `1.0.0.dev0`, under active development on the `v1.0-rewrite` branch. Not yet on PyPI.
> v1.0 replaces the 0.2 code, whose methods had the problems documented in
> [`docs/audit/notebooks-v1.2.md`](docs/audit/notebooks-v1.2.md).

## Install

```bash
pip install "git+https://github.com/DaneshSelwal/reguq@v1.0-rewrite"                    # core: numpy, scipy, pandas, scikit-learn
pip install "reguq[boosting,ngboost,tuning] @ git+https://github.com/DaneshSelwal/reguq@v1.0-rewrite"
pip install "reguq[all] @ git+https://github.com/DaneshSelwal/reguq@v1.0-rewrite"       # everything, incl. torch
```

`import reguq` never imports torch or any boosting backend. Each backend is loaded only by the
estimator that needs it. Extras: `boosting`, `gpboost`, `ngboost`, `pgbm`, `torch`, `tabnet`,
`qrf`, `treeffuser`, `mapie`, `puncc`, `tuning`, `report`, `explain`, `all`.

## Quick start

```python
from lightgbm import LGBMRegressor
from reguq.conformal import ConformalizedQuantileRegressor, SplitConformalRegressor
from reguq.metrics import interval_report

# 90 % split-conformal interval; the calibration split is held out automatically
cp = SplitConformalRegressor(LGBMRegressor(), calib_size=0.25, random_state=0).fit(X_train, y_train)
interval = cp.predict_interval(X_test, alpha=0.1)          # shape (n, 2)
print(interval_report(y_test, interval, alpha=0.1))         # coverage, width, Winkler score, ...

# CQR with real alpha/2 and 1-alpha/2 quantile models
cqr = ConformalizedQuantileRegressor(LGBMRegressor(), alpha=0.1).fit(X_train, y_train)
```

For time series, pass `time_ordered=True`: nothing is shuffled, and the calibration block is the end
of the training period. The online methods are strictly causal:

```python
from reguq.conformal import OnlineConformalRegressor
aci = OnlineConformalRegressor(LGBMRegressor(), method="aci", alpha=0.1).fit(X_train, y_train)
intervals = aci.predict_interval_online(X_test, y_test)    # interval at t uses only data before t
```

## Methods

| Family | Class | Guarantee | Reference |
|---|---|---|---|
| Split conformal (absolute / normalized) | `conformal.SplitConformalRegressor` | marginal ≥ 1−α, ≤ 1−α+1/(n+1) | Lei et al. 2018 |
| CQR (symmetric / per-tail) | `conformal.ConformalizedQuantileRegressor` | marginal ≥ 1−α | Romano et al. 2019 |
| CV+ / Jackknife+ / minmax | `conformal.CrossConformalRegressor` | ≥ 1−2α (plus), ≥ 1−α (minmax) | Barber et al. 2021 |
| Jackknife+-after-bootstrap | `conformal.JackknifePlusAfterBootstrapRegressor` | ≥ 1−2α | Kim, Xu & Barber 2020 |
| NexCP (weighted, non-exchangeable) | `conformal.NexCPRegressor` | ≥ 1−α − drift term | Barber et al. 2023 |
| ACI | `OnlineConformalRegressor(method="aci")` | long-run error → α for any sequence | Gibbs & Candès 2021 |
| DtACI / FACI | `method="dtaci"` | strongly adaptive regret | Gibbs & Candès 2024 |
| SF-OGD, SAOCP | `method="sf_ogd"`, `"saocp"` | adaptive regret | Bhatnagar et al. 2023 |
| Online NexCP | `method="nexcp"` | as NexCP | Barber et al. 2023 |
| POT / GPD tail extrapolation | `conformal.ExtremeValueConformalRegressor` | **none** (semi-parametric) | Coles 2001 |
| Adaptive coverage policies | `conformal.AdaptiveCoveragePolicyRegressor` | post-hoc E[P(miss)/α̃] ≤ 1 | Gauthier, Bach & Jordan 2026 |
| Multi-quantile + rearrangement | `quantile.MultiQuantileRegressor` | none (wrap in CQR) | Chernozhukov et al. 2010 |
| Quantile regression forest | `quantile.QuantileForestRegressor` | none (wrap in CQR) | Meinshausen 2006 |
| NGBoost, PGBM | `probabilistic.NGBoostRegressor`, `PGBMRegressor` | model-based | Duan et al. 2020; Sprangers et al. 2021 |
| IBUG | `probabilistic.IBUGRegressor` | model-based | Brophy & Lowd 2022 |
| CARD diffusion | `probabilistic.CARDRegressor` | model-based | Han, Zheng & Zhou 2022 |
| HCM | `probabilistic.HCMRegressor` | model-based | Choi et al. 2026 |
| Treeffuser | `probabilistic.TreeffuserRegressor` | model-based | Beltran-Velez et al. 2024 |

A model-based distribution can be given a finite-sample guarantee by using its predicted std as the
scale of a normalized split-conformal score (`probabilistic.StdEstimator`).

## Metrics

`reguq.metrics`:
- **Interval:** coverage (PICP), MPIW, PINAW, the Winkler interval score, CWC, binned and
  size-stratified coverage.
- **Quantile and distributional:** pinball loss, CRPS (Gaussian, sample and quantile-grid), Gaussian
  NLL, PIT, calibration error and ENCE.
- **Point:** RMSE, MAE, R² and bias.

## Pipeline

```bash
reguq-run config.yaml
```

The config sets the train/test CSVs, target, `time_ordered`, `alpha`, models, methods and tuning.
Hyper-parameters are tuned **on the training data only** (cross-validation or forward-chaining) and
shared by every method. The test set is touched once, for evaluation. A method that fails is
reported with its error message; it is never silently replaced by another method.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -m "not slow"          # statistical tests are marked `statistical`
```

## License

MIT
