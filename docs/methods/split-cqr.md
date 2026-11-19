# Split conformal and CQR

## Split conformal: `SplitConformalRegressor`

1. Split the training data into a proper training part and a calibration part.
2. Fit \(\hat\mu\) on the proper training part.
3. Compute scores on the calibration part and take \(\hat q\), the
   [conformal quantile](conformal-foundations.md).

=== "absolute score (`score='absolute'`)"

    \[
    S_i=|Y_i-\hat\mu(X_i)|,\qquad \hat C(x)=[\hat\mu(x)-\hat q,\ \hat\mu(x)+\hat q].
    \]

    Every point gets the same width.

=== "normalized score (`score='normalized'`)"

    \[
    S_i=\frac{|Y_i-\hat\mu(X_i)|}{\hat\sigma(X_i)},\qquad
    \hat C(x)=[\hat\mu(x)-\hat q\,\hat\sigma(x),\ \hat\mu(x)+\hat q\,\hat\sigma(x)].
    \]

    \(\hat\sigma\) is fitted to **out-of-fold** absolute residuals on the proper training part.
    In-sample residuals of a flexible model are almost zero and would make \(\hat\sigma\) useless.
    \(\hat\sigma\) is floored at \(10^{-3}\times\) its median, to avoid dividing by zero.
    Any fitted model with `predict` can be passed as `scale_estimator`, including
    `probabilistic.StdEstimator` together with `prefit=True`.

**Guarantee** (Papadopoulos et al. 2002; Lei et al. 2018): for exchangeable data,
\(1-\alpha\le P(Y\in\hat C(X))\le 1-\alpha+1/(n_{\text{cal}}+1)\). The guarantee holds whatever
\(\hat\mu\) and \(\hat\sigma\) are. A better model only gives narrower intervals.

## Conformalized quantile regression: `ConformalizedQuantileRegressor`

Romano, Patterson & Candès (2019). Fit two quantile models, \(\hat q_{\alpha/2}\) and
\(\hat q_{1-\alpha/2}\), with the pinball loss (see [quantile regression](quantile.md)), then

\[
S_i=\max\{\hat q_{\alpha/2}(X_i)-Y_i,\ Y_i-\hat q_{1-\alpha/2}(X_i)\},\qquad
\hat C(x)=[\hat q_{\alpha/2}(x)-\hat q,\ \hat q_{1-\alpha/2}(x)+\hat q].
\]

`S_i` is negative when the point lies inside the band, so CQR can **shrink** a band that is too wide.

- The `alpha` passed to the constructor sets the quantile levels the models are trained at. Calling
  `predict_interval` with a different α refits nothing and raises an error.
- `symmetric=False` calibrates each tail separately at level α/2 (Romano et al. 2019, Thm 2). This
  gives the same marginal guarantee and better balance between the tails, at the cost of wider
  intervals when \(n\) is small.
- The base estimator must support a quantile objective. `reguq.quantile.with_quantile` sets it for
  scikit-learn GBR/HGB, `QuantileRegressor`, LightGBM, XGBoost (`reg:quantileerror`), CatBoost
  (`Quantile:alpha=…`), GPBoost and `TabNetRegressor`. Other estimators raise a `TypeError`. You can
  also pass `lower_estimator` and `upper_estimator` explicitly.
- If the quantile models cross, the interval ends are re-ordered with min/max.

!!! danger "v1.2 notebook error"
    The old "CQR" built both bounds from the **same mean regressor**, so it was split conformal under
    another name (audit MP-2).

## References
Lei et al. 2018; Papadopoulos et al. 2002; Romano, Patterson & Candès 2019; Vovk, Gammerman &
Shafer 2005. See the [bibliography](../bibliography.md).
