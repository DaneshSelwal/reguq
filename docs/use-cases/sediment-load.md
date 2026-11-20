# Use case: daily sediment load

Notebook: [`examples/use_cases/sediment_load.ipynb`](https://github.com/DaneshSelwal/reguq/blob/main/examples/use_cases/sediment_load.ipynb)

## Problem

Predict the suspended sediment load \(S_t\) from discharge \(Q_t, Q_{t-1}\) and \(S_{t-1}\), with
**90 % prediction intervals** (α = 0.1).

| | Train (3 years) | Test (following year) |
|---|---|---|
| Days | 1095 | 365 |
| Mean / SD of \(S_t\) | 992 / 4 860 | 369 / 1 235 |
| Median / max | 47 / 84 400 | 23 / 7 400 |

The training years contain extreme floods that the test year does not, which makes this a
**regime change**. All runs use `time_ordered=True`, so nothing is shuffled and calibration blocks are
taken from the end of the training period.

## Protocol

- **Tuning:** 6 models (LightGBM, XGBoost, CatBoost, gradient boosting, random forest, GPBoost), each
  tuned with Optuna on the **training years only**. TPE sampler, median pruner, 40 trials, 5-fold
  forward-chaining CV; 12–19 trials pruned per model.
- **Methods:** 14 interval methods per model, plus NGBoost, QRF, HCM, Treeffuser and CARD (with the
  LightGBM mean model).
- **Evaluation:** a single evaluation on the 365 test days. The Monte-Carlo standard error of a
  coverage estimate is \(\sqrt{0.1\cdot0.9/365}\approx0.016\).
- **Environment:** Kaggle CPU, Python 3.13, scikit-learn 1.6.1, reguq `main@6ff8ccc`, about 15 min.

## Results by method

Averaged over the models each method ran with, and ranked by the median interval score (lower is better).

| Method | Models | Mean coverage | Range | Median width | Median interval score |
|---|---|---|---|---|---|
| `treeffuser` | 1 | 0.948 | 0.948–0.948 | 441 | 913 |
| `normalized` | 6 | 0.818 | 0.715–0.910 | 589 | 1,011 |
| `cqr` | 5 | 0.785 | 0.701–0.830 | 813 | 1,021 |
| `multi_quantile` | 5 | 0.830 | 0.756–0.868 | 793 | 1,029 |
| `ngboost` | 1 | 0.934 | 0.934–0.934 | 343 | 1,126 |
| `hcm` | 1 | 0.871 | 0.871–0.871 | 423 | 1,137 |
| `ibug` | 5 | 0.961 | 0.945–0.978 | 849 | 1,190 |
| `dtaci` | 6 | 0.900 | 0.879–0.912 | 604 | 1,451 |
| `card` | 1 | 0.973 | 0.973–0.973 | 1,469 | 1,883 |
| `nexcp_online` | 6 | 0.883 | 0.841–0.907 | 700 | 2,136 |
| `sf_ogd` | 6 | 0.901 | 0.899–0.901 | 1,821 | 2,151 |
| `jab` | 6 | 0.943 | 0.926–0.959 | 1,056 | 2,268 |
| `aci` | 6 | 0.889 | 0.879–0.899 | 575 | 2,279 |
| `cv_plus` | 6 | 0.962 | 0.929–0.986 | 1,709 | 2,333 |
| `split` | 6 | 0.863 | 0.778–0.893 | 344 | 2,373 |
| `nexcp` | 6 | 0.837 | 0.663–0.896 | 372 | 2,392 |
| `saocp` | 6 | 0.959 | 0.918–0.989 | 2,289 | 2,533 |
| `acp` | 6 | 0.979 | 0.964–0.997 | 2,569 | 3,442 |

## Ten best runs by interval score

| Method | Model | Coverage | Mean width | Interval score | RMSE | R² |
|---|---|---|---|---|---|---|
| `cqr` | xgboost | 0.811 | 515 | 666 | 369 | 0.910 |
| `multi_quantile` | xgboost | 0.868 | 543 | 680 | 392 | 0.899 |
| `multi_quantile` | gradientboosting | 0.756 | 531 | 760 | 418 | 0.885 |
| `cqr` | gradientboosting | 0.830 | 504 | 809 | 330 | 0.928 |
| `normalized` | gradientboosting | 0.910 | 598 | 823 | 623 | 0.745 |
| `treeffuser` | - | 0.948 | 441 | 913 | 662 | 0.712 |
| `normalized` | lightgbm | 0.797 | 409 | 965 | 407 | 0.891 |
| `normalized` | xgboost | 0.805 | 459 | 977 | 596 | 0.766 |
| `ibug` | xgboost | 0.945 | 755 | 1,019 | 713 | 0.666 |
| `cqr` | lightgbm | 0.822 | 813 | 1,021 | 401 | 0.894 |

## Runs that did not complete

| Method | Model | Status |
|---|---|---|
| `cqr` | randomforest | error: TypeError: don't know how to make RandomForestRegressor predict a quantile; pass explicit lower/upper quantile es |
| `multi_quantile` | randomforest | error: TypeError: don't know how to make RandomForestRegressor predict a quantile; pass explicit lower/upper quantile es |
| `ibug` | gpboost | error: TypeError: IBUG needs a tree ensemble exposing leaf indices; GPBoostRegressor is not supported (use LightGBM, XGB |
| `pgbm` | - | error: ImportError: pgbm.sklearn is required here: pip install 'reguq[pgbm]' |
| `qrf` | - | error: TypeError: _get_n_samples_bootstrap() takes 2 positional arguments but 3 were given |

The first three combinations are impossible by design: random forest has no quantile objective,
and IBUG cannot read GPBoost's leaves. They are reported as errors and never replaced by another
method. `pgbm` is not available on Python ≥ 3.12. The `qrf` failure is a version mismatch on
Kaggle: quantile-forest 1.5 needs scikit-learn ≥ 1.9.

## Interpretation

1. **The online methods hit the target.** ACI (0.879–0.899), DtACI (0.879–0.912) and SF-OGD
   (0.899–0.901) are within about one standard error of 0.90 for **every** model. Their
   long-run guarantee does not require exchangeability, and this data violates exchangeability.
2. **Static methods undercover after the regime change.** Split conformal (0.78–0.89), the normalized
   score (0.72–0.91) and CQR (0.70–0.83) are calibrated on the end of a flood period. Their good
   interval scores reflect narrow intervals, not validity.
3. **Resampling methods are conservative.** CV+ (0.93–0.99) and J+aB (0.93–0.96) use residuals from
   the whole training period, including the floods.
4. **ACP** kept \(\tilde\alpha = 0.1\), as intended by the default \(\lambda_\alpha\). Its mean-based
   (e-value) interval is wide (coverage 0.96–1.00) by construction. See [ACP](../methods/acp.md).
5. **Model-based distributions:** NGBoost (0.934, CRPS 109) and Treeffuser (0.948, CRPS 99) are reasonably
   calibrated here. HCM gives the best point predictions (R² 0.936) but undercovers (0.871).
6. The errors are dominated by a few flood days. A log-transformed target is the natural next experiment.

!!! note "No comparison with the v1.2 notebooks"
    The saved outputs in the v1.2 `sediment_load` folder come from the concrete and noise datasets, not
    from sediment load (audit D5), so no earlier sediment-load numbers exist to compare against.
