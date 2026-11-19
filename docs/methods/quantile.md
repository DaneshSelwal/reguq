# Quantile regression

## Pinball loss

A model \(\hat q_\tau\) estimates the conditional \(\tau\)-quantile by minimizing

\[
\rho_\tau(u)=\max\{\tau u,\ (\tau-1)u\},\qquad u=y-\hat q_\tau(x)
\]

(Koenker & Bassett 1978). `metrics.pinball_loss` computes it.

## Quantile objectives: `with_quantile`

`reguq.quantile.with_quantile(estimator, tau)` returns a clone set to the τ-quantile objective:

| Estimator | Setting |
|---|---|
| sklearn `GradientBoostingRegressor` | `loss="quantile", alpha=τ` |
| sklearn `HistGradientBoostingRegressor` | `loss="quantile", quantile=τ` |
| sklearn `QuantileRegressor` | `quantile=τ` (linear, Koenker–Bassett) |
| LightGBM | `objective="quantile", alpha=τ` |
| XGBoost ≥ 2.0 | `objective="reg:quantileerror", quantile_alpha=τ` |
| CatBoost | `loss_function="Quantile:alpha=τ"` |
| GPBoost | `objective="quantile", alpha=τ` |
| `reguq.quantile.TabNetRegressor` | pinball loss (see below) |
| any object with `set_quantile(τ)` | user-defined |

An estimator without a quantile objective, such as a random forest, raises a `TypeError`. It is never
silently replaced by a mean model, which was the cause of audit MP-2.

## Several quantiles without crossing: `MultiQuantileRegressor`

Fitting each τ separately can give \(\hat q_{\tau_1}(x)>\hat q_{\tau_2}(x)\) for \(\tau_1<\tau_2\).
**Rearrangement** (Chernozhukov, Fernández-Val & Galichon 2010) sorts the predicted quantiles at
each x:

\[
\tilde q_{\tau_{(1)}}(x)\le\dots\le\tilde q_{\tau_{(K)}}(x)\quad=\quad\mathrm{sort}\{\hat q_{\tau_k}(x)\}_k .
\]

Rearrangement never increases the error with respect to the true quantile curve, in any \(L^p\) norm.
`crossing_rate` reports how often the raw models crossed. `predict_quantiles(X)` returns an
`(n, K)` array, and `predict_interval(X, alpha)` uses the levels α/2 and 1−α/2, which must be in the
grid.

On its own this gives **no coverage guarantee**. For one, wrap the base estimator in
[CQR](split-cqr.md).

## Quantile regression forest: `QuantileForestRegressor`

Meinshausen (2006). One random forest gives the conditional distribution as weights on training
targets: \(\hat F(y\mid x)=\sum_i w_i(x)\mathbb 1\{Y_i\le y\}\), where \(w_i(x)\) averages over trees
the share of \(x\)'s leaf that point \(i\) occupies. Quantiles are read from \(\hat F\). This uses the
`quantile-forest` package. It is not the spread of tree means, which was the error in reguq 0.2.

## TabNet: `TabNetRegressor`

An sklearn-compatible wrapper around `pytorch-tabnet` (Arik & Pfister 2021). With `quantile=τ`, it
trains on the pinball loss. Early stopping uses a **held-out** validation split (`val_size`,
time-ordered if requested) with the pinball metric. The v1.2 notebook early-stopped on the training
MSE (audit Q3).
