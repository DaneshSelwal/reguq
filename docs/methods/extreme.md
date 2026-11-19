# Extreme-value tails (POT/GPD)

`ExtremeValueConformalRegressor` and `gpd_tail_quantile`.

When α is small relative to \(n_{\text{cal}}\), the conformal quantile becomes a single extreme order
statistic (or \(+\infty\) when \(\alpha<1/(n+1)\)). Peaks-over-threshold extrapolation smooths the
tail with a generalized Pareto distribution (GPD; Pickands 1975; Coles 2001).

## Formula

Choose a threshold \(u\), the `threshold` empirical quantile of the calibration scores (default
0.9). Let \(N_u\) be the number of scores above \(u\), out of \(n\). Fit a GPD with shape ξ and scale
σ to the exceedances \(S_i-u>0\). Then for \(s>u\),

\[
P(S>s)\approx\frac{N_u}{n}\Big(1+\xi\frac{s-u}{\sigma}\Big)^{-1/\xi}.
\]

Setting this equal to α and solving gives

\[
\hat q=u+\frac{\sigma}{\xi}\Big[\Big(\frac{n\alpha}{N_u}\Big)^{-\xi}-1\Big]\qquad(\xi\neq0),\qquad
\hat q=u-\sigma\log\frac{n\alpha}{N_u}\qquad(\xi=0).
\]

The interval is \(\hat\mu(x)\pm\hat q\), symmetric because the score is \(|y-\hat\mu|\). Coverage is
therefore evaluated **two-sided**.

## Fallbacks and guarantee

- If \(\alpha\ge N_u/n\), the target quantile is below the threshold. The ordinary conformal quantile
  is then used. `gpd_tail_quantile(model.calib_scores_, alpha)` returns the fitted
  `u, xi, sigma, n_exceed` and `method` (`"gpd"` or `"conformal"`), so you can inspect which one applied.
- If there are fewer than `min_exceedances` (default 10) exceedances, the result also falls back to
  the conformal quantile.

**There is no finite-sample guarantee.** The result is a semi-parametric estimate that is valid
asymptotically if the GPD tail model holds. Treat it as an extrapolation, and compare its coverage
on held-out data.

!!! danger "v1.2 notebook error"
    The old code used \(u+\frac{\sigma}{\xi}\big((n/(n\alpha))^{\xi}-1\big)\). This drops the factor
    \(N_u/n\) and gave interval widths up to \(1.8\times10^{6}\) (audit NX-10).
