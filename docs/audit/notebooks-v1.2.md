# Audit of the v1.2 regression-UQ notebooks

**Scope.** The 8 canonical notebooks in
`DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/Regression/<phase>/` and the 72
per-dataset copies in `examples/<dataset>/<phase>/` (9 datasets × 8 phases), plus the draft
library `reguq` 0.2.0. This was a read-only review: every code cell was read and the saved outputs
were inspected. No notebook was re-executed, and the original repository was not modified.

**Main conclusion.** The per-dataset copies share the canonical notebooks' math code (they differ
only in paths, feature names and Excel helpers), so **every issue below affects all 9 datasets**.
Several issues are critical and invalidate published numbers, in particular all MAPIE coverages, all
CARD results, all Treeffuser results, the NEXCP/online/MFCS/COP intervals and every test-set metric
that depends on the Optuna hyper-parameters.

Severity: **C** = critical (invalidates results), **M** = major (biases results / wrong
label), **m** = minor.
Cell numbers are 0-based indices of code cells.

---

## 0. Cross-cutting

| ID | Sev | Issue | Effect | Fixed in v1.0 by |
|---|---|---|---|---|
| F1 | C | `Optuna_autosampler` cell 16 scores each trial with `mean_squared_error(y_test, model.predict(X_test))`. The best model is the minimum over 8 runs × 50 trials of **test** MSE, and every other phase loads these parameters. | Test RMSE/R² are optimistically biased, and UQ is evaluated on data used for model selection. | `reguq.tuning.tune`: CV on training data only |
| F2 | M | The objectives never call `trial.report`/`should_prune`, so Median/Hyperband/Wilcoxon/Threshold pruners can never prune. `AutoSampler()` is unseeded. | The "pruner comparison" is 8 unseeded repeats, so its ranking is noise. | per-fold `trial.report`; seeded samplers |
| D1 | M | Time-series data (sediment load: Qt, Qt-1, St-1 → St, daily) is split with `shuffle=True` / `KFold(shuffle=True)` in several places. | Exchangeability assumptions are violated and results are not reproducible. | `time_ordered=True` everywhere (`reguq.splits`) |
| D2 | M | **EWQI dataset**: train target `Pre monsoon EWQI`, test target `Post monsoon EWQI`. | The test measures a different quantity (pre→post-monsoon transfer), not test error. | pipeline raises unless `test_target` is set explicitly |
| D3 | m | Column names contain stray whitespace (`"Q "`, `" Skew  "`). "streamflow_uncertainty_analysis" contains water-quality variables (pH, salinity, turbidity, temperature → TDS). | Fragile column selection; misleading dataset name. | pipeline strips names |
| D4 | m | α differs between notebooks (0.05 in MAPIE and NEXCP, 0.1 in PUNCC and ACP), and so do the CWC definitions. | Methods are not compared like-for-like. | one `alpha` per pipeline run; one `metrics` module |

## 1. Hyper-parameter tuning (`Optuna_autosampler.ipynb`)

| ID | Sev | Issue |
|---|---|---|
| F1, F2 | C, M | see §0 |
| F3 | m | Cell 21 passes unscaled `x_test` to LIME/SHAP for models trained on scaled data. |
| F4 | m | Cell 19 retrains from `best_params`, so its results differ from the saved model (TabNet and PGBM are non-deterministic). The GBR space includes `loss='quantile'` for a point model. |
| F5 | m | In all 9 example copies `model_classes` was removed but is still used, which raises `NameError` in the interpretability step. |

## 2. Quantile regression (`Quantile_Regression.ipynb`)

| ID | Sev | Issue | Correct |
|---|---|---|---|
| Q1 | M | Cell 29: XGBoost quantile models are trained with **LightGBM's** `best_params` (`best_paramst` is computed but never used). | use the XGBoost parameters (`reguq.models` + one params dict per model) |
| Q2 | M | Quantile crossing is not handled. TabNet q0.2 = 5.8 < q0.1 = 19.7; HGBR q0.85 > q0.9. This gives negative widths and sign-flipped conformal scores. | monotone rearrangement (Chernozhukov et al. 2010): `reguq.quantile.rearrange_quantiles` |
| Q3 | M | TabNet quantile models early-stop on **training** MSE (`eval_set=[(X_train, y_train)]`). | held-out split with a pinball metric: `reguq.quantile.TabNetRegressor` |
| Q4 | M | CRPS = `crps_ensemble` over 13 unevenly spaced quantiles, as if they were equally weighted samples. | CRPS = 2∫ρ_τ dτ with grid weights: `reguq.metrics.crps_quantile` |
| Q5 | C | `n = total_samples / 2` (float) crashes in the canonical notebook. Calibration is a random split of a time-series test set, and "after calibration" coverage is measured on half the test set. | a calibration set taken from training data |
| Q6 | m | The "conformal" plot is calibrated in-sample on the full test set without the (n+1) correction. | — |
| Q7 | m | NGBoost μ is constant (29.27) on every row; PGBM gives negative lower quantiles for a non-negative target. | — |

## 3. Probabilistic distribution (`Probabilistic__Distribution.ipynb`)

| ID | Sev | Issue | Correct |
|---|---|---|---|
| P1 | M | The CRPS decomposition gives Reliability ≈ total CRPS and Resolution ≈ Uncertainty for every model (≈ 543.6), so it carries no information. | Hersbach (2000) / CORP decomposition (not shipped in v1.0) |
| P2 | M | IBUG: `std = scale ** 0.5`. IBUG's `scale` is already a std (reference code: `np.std(train_vals)`), so σ is shrunk to ≈ 7–8, which is severely overconfident (observed 0.45–0.69 at nominal 0.1). | `reguq.probabilistic.IBUGRegressor` (native, std not square-rooted) |
| P3 | M | Cell 78 writes the LightGBM-IBUG plots and results into `XGB_Prob.xlsx`. | — |
| P4 | C | Treeffuser outputs are degenerate: GBR mean −920 / sd 8180 and XGB mean −1806 / sd 10440, for targets ≈ 30. Point-model hyper-parameters were fed to the score model. | `TreeffuserRegressor` uses its own parameters |
| P5 | M | The "calibration" panel compares y with percentiles of **one pooled** random draw across all points (unseeded). That is not the per-point PIT. | `reguq.metrics.pit_gaussian` / `calibration_curve` |
| P6 | m | The calibration curve includes the trivial levels 0 and 1. A Gaussian is assumed for a heavy-tailed target. | levels 0.05…0.95 |

## 4. CARD (`Probabilistic__Distribution(CARD).ipynb`)

| ID | Sev | Issue | Correct |
|---|---|---|---|
| C1 | C | `CARDRegressor` (cell 21) is not a diffusion model. There is no noise schedule, and no noisy input y_t is ever built, so the network sees neither residuals nor y_t and its optimum is ε̂ ≈ 0. Prediction is a single step at t = 0 giving `samples = ŷ + N(0,1)`, i.e. **σ ≈ 1 in raw target units for every model and dataset** (outputs: 0.89–1.12). | CARD (Han et al. 2022): forward process to f(x), ε-network on (x, y_t, f(x), t), reverse posterior with γ₀, γ₁, γ₂: `reguq.probabilistic.CARDRegressor` |
| C2 | m | The GPBoost base prediction is constant (87.27); the TabNet base gives negative means. | — |

## 5. HCM (`Hyperspherical_Confidence_Mapping(HCM).ipynb`)

| ID | Sev | Issue |
|---|---|---|
| H1 | M | The canonical notebook loads the **noise** dataset and silently falls back to synthetic demo data if a CSV is missing. |
| H2 | M | The model collapsed to the mean (constant 67.4); corr(σ, abs error) = −0.32. Structural reason: with standardised scalar y, the direction (y, y)/‖·‖ only encodes sign(y), so ‖d̂‖ mostly reflects sign ambiguity near the mean. |
| H3 | m | `ECE_reg = abs(mean σ − MAE)` compares unlike quantities (E\|e\| = 0.798σ for a Gaussian). | 
| H4 | m | The same split is used for early stopping and temperature scaling. |

v1.0: `HCMRegressor` follows Choi et al. (ICLR 2026), with separate early-stopping and temperature
splits and the limitation documented; `reguq.metrics.ence` replaces ECE_reg.

## 6. MAPIE / PUNCC (`Conformal Predictions(MAPIE,PUNCC).ipynb`)

| ID | Sev | Issue | Correct |
|---|---|---|---|
| MP-1 | C | Coverage is computed as `(y >= int_pred[:, 0]) & (y <= int_pred[:, 1])` where `int_pred[:, 0]` has shape (n, 1). This broadcasts to an n×n matrix, so the printed "coverage" is the share of all (point, interval) pairs. **Every printed MAPIE coverage, CWC and conditional-coverage number is invalid.** The `is_covered` column in Excel is correct. | `int_pred[:, 0, 0]`; `reguq.metrics.coverage` refuses to broadcast |
| MP-2 | C | "CQR" builds the lower and upper models from the **same** mean regressor, so the result equals split CP with absolute residuals. | quantile objectives at α/2 and 1−α/2 (Romano et al. 2019): `ConformalizedQuantileRegressor` |
| MP-3 | M | "SCP" = MAPIE `method='base', cv=KFold(5)`, which is CV, not split. "J+aB" = `method='minmax'`, which is J-minmax-aB. Unshuffled KFold on time-ordered data. | `SplitConformalRegressor`, `JackknifePlusAfterBootstrapRegressor` |
| MP-4 | M | `PGBMWrapper.__init__(**params)` is not clone-safe, so MAPIE PGBM runs silently use default parameters. | explicit `__init__` parameters (all v1.0 estimators pass `sklearn.clone`) |
| MP-5 | m | CVPlus plots/exports the split-CP prediction; GradientBoosting is labelled "GPBoost"; CatBoost PUNCC runs without parameters. | — |

## 7. NEXCP / online (`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`)

| ID | Sev | Issue | Correct |
|---|---|---|---|
| NX-1 | C | NexCP-Split, Online-Split, Online-CV+, SAOCP, FACI, SF-OGD, MFCS-Split/Full and COP use **in-sample training residuals** as conformity scores. There is no calibration set, so there is no guarantee. | scores on held-out calibration data |
| NX-2 | M | The NexCP weighted quantile normalises over calibration weights only, omits the test-point mass at +∞, and interpolates linearly. | Barber et al. (2023) eq. 5: `weighted_conformal_quantile` |
| NX-3 | C | "NexCP-Full" = an unweighted in-sample quantile (identical to Online-Split and MFCS-Split: 1925.006158). "J+aB" uses in-bag residuals. "CV+" averages per-fold endpoints. None of the three is the named method. | `CrossConformalRegressor` (weights optional), `JackknifePlusAfterBootstrapRegressor` |
| NX-4 | C | SF-OGD: the gradient has the wrong sign (widens when covered), the interval at t is built after updating with y_t (look-ahead), and the step is not scale-free. | `ScaleFreeOGD` (port of Bhatnagar et al. 2023) |
| NX-5 | C | "FACI" chooses its window using the loss that includes y_t (look-ahead), and is a window selector rather than FACI. | `DtACI` (Gibbs & Candès 2024) |
| NX-6 | M | "SAOCP" = a fixed 500-point window. | `SAOCP` (Bhatnagar et al. 2023) |
| NX-7 | C | "MFCS-Split" ≡ Online-Split (identical digits). MFCS is not defined anywhere. | removed: undefined method |
| NX-8 | M | Adaptive Conformal Inference is never implemented although the notebook title claims it. | `ACI` (Gibbs & Candès 2021) |
| NX-9 | M | The COP learning rate (0.05) is not scaled to residuals ≈ 1000, so the result is effectively the static quantile. | removed: undefined method (no reference) |
| NX-10 | C | Extreme/GPD CP: `q = u + σ/ξ((n/(nα))^ξ − 1)` drops N/N_u, giving widths up to 1.8 M and 100 % coverage. Coverage is one-sided against a two-sided interval. | `u + σ/ξ[(nα/N_u)^(−ξ) − 1]`: `gpd_tail_quantile`; two-sided coverage |
| NX-11 | C | AlphaNet: the size term (target units ≈ 600) dominates λ = 20, so α̃ → 1 always. The leave-one-out does not refit, and the input is unscaled. | see ACP below |

## 8. ACP (`Adaptive_Coverage_Policies(ACP).ipynb`)

| ID | Sev | Issue | Correct |
|---|---|---|---|
| ACP-1 | C | The target α is ignored (`run_full_acp_pipeline` has no α). The scale problem of NX-11 pushes α̃ to the upper clip. | λ in target units: default λ_α = 2T̄n/(nα−1)², or bisection to a target size (Gauthier, Bach & Jordan 2026, Alg. 2) |
| ACP-2 | M | λ is selected by comparing training losses that include λ·α̃, so the smallest λ always wins. | — |
| ACP-3 | M | A sigmoid saturating near 0 makes the denominator negative and the width blows up (×1e6). | parametrise α̃ ∈ (1/n, 1) |
| ACP-4 | m | In the .md worked example, ceil(6·0.9) = 6 > n = 5 should give an infinite interval. | `conformal_quantile` returns +∞ |

The guarantee ACP actually provides is **post-hoc**, E[P(miss | α̃)/α̃] ≤ 1 (Prop. 2.2), not
marginal 1 − α coverage. v1.0 states this explicitly.

## 9. Draft library `reguq` 0.2.0

The draft was not published on PyPI, and its CI had been failing since 2026-07 (a module-level
`import torch`). Its dependencies were pinned exactly with `==`. Every PUNCC adapter raised and fell
back silently to split CP under the original label. The NEXCP/online methods copied the in-sample
residuals. ACI was not implemented. The RF "quantiles" were tree-mean spread, σ fell back to a
homoscedastic in-sample residual std, and the target scaler was discarded, so outputs were in
standardised units. Stale Optuna checkpoints were resumed across runs, and the tests were
smoke-only (`coverage >= 0.0`). v1.0 replaces the core; the design decisions are documented in the
module docstrings.

## References

- Angelopoulos & Bates (2023). A gentle introduction to conformal prediction. *FnT ML*.
- Barber, Candès, Ramdas & Tibshirani (2021). Predictive inference with the jackknife+. *Ann. Stat.*
- Barber, Candès, Ramdas & Tibshirani (2023). Conformal prediction beyond exchangeability. *Ann. Stat.*
- Bhatnagar, Wang, Xiong & Bai (2023). Improved online conformal prediction via strongly adaptive online learning. *ICML*.
- Brophy & Lowd (2022). Instance-based uncertainty estimation for gradient-boosted regression trees. *NeurIPS*.
- Chernozhukov, Fernández-Val & Galichon (2010). Quantile and probability curves without crossing. *Econometrica*.
- Choi et al. (2026). Uncertainty estimation via hyperspherical confidence mapping. *ICLR*.
- Gauthier, Bach & Jordan (2026). Adaptive coverage policies in conformal prediction. *AISTATS*.
- Gibbs & Candès (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
- Gibbs & Candès (2024). Conformal inference for online prediction with arbitrary distribution shifts. *JMLR*.
- Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
- Han, Zheng & Zhou (2022). CARD: Classification and regression diffusion models. *NeurIPS*.
- Kim, Xu & Barber (2020). Predictive inference is free with the jackknife+-after-bootstrap. *NeurIPS*.
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman (2018). Distribution-free predictive inference for regression. *JASA*.
- Romano, Patterson & Candès (2019). Conformalized quantile regression. *NeurIPS*.
