# Metrics

All metrics are in `reguq.metrics`. They take 1-D arrays and **refuse to broadcast** `(n, 1)`
against `(n,)`, which was the cause of the n×n MAPIE coverage bug (audit MP-1).

## Intervals

| Function | Definition | Better |
|---|---|---|
| `coverage` (PICP) | \(\frac1n\sum\mathbb 1\{l_i\le y_i\le u_i\}\) | close to 1−α |
| `mean_width` (MPIW) | \(\frac1n\sum(u_i-l_i)\) | lower, at valid coverage |
| `normalized_mean_width` (PINAW) | MPIW / \((\max y-\min y)\) | lower |
| `interval_score` (Winkler) | see below | **lower; proper** |
| `coverage_width_criterion` (CWC) | \(\text{PINAW}\,(1+\mathbb 1\{\text{PICP}<1-\alpha\}e^{-\eta(\text{PICP}-(1-\alpha))})\), η = 50 | lower; not proper |
| `binned_coverage` | coverage per quantile bin of width (or of any `by`) | flat at 1−α |
| `size_stratified_coverage` (SSC) | worst-bin coverage | close to 1−α |

The **interval score** (Winkler 1972; Gneiting & Raftery 2007, §6.2) is a proper scoring rule for
the central \((1-\alpha)\) interval:

\[
\mathrm{IS}_\alpha(l,u;y)=(u-l)+\frac2\alpha(l-y)\mathbb 1\{y<l\}+\frac2\alpha(y-u)\mathbb 1\{y>u\}.
\]

Use it to rank methods. CWC (Khosravi et al. 2011) is provided only for comparison with the
literature: it is not proper, and its value depends on the arbitrary η.

## Quantiles and distributions

**Pinball loss:** \(\rho_\tau(y-q)\); see [quantile regression](quantile.md).

**CRPS**, \(\mathrm{CRPS}(F,y)=\int(F(z)-\mathbb 1\{y\le z\})^2dz\), lower is better and proper:

- `crps_gaussian`: closed form,
  \(\sigma\big[z(2\Phi(z)-1)+2\varphi(z)-\tfrac1{\sqrt\pi}\big]\) with \(z=(y-\mu)/\sigma\).
- `crps_sample`: \(\mathbb E|X-y|-\tfrac12\mathbb E|X-X'|\), computed in \(O(m\log m)\) from sorted
  samples. With `fair=True`, it uses the unbiased \(m(m-1)\) denominator (Ferro 2014).
- `crps_quantile`: \(2\int_0^1\rho_\tau(y-F^{-1}(\tau))d\tau\) on a grid of quantiles, weighted by
  `quantile_grid_weights`. **Unevenly spaced levels must be weighted**; treating them as an
  equally weighted ensemble was audit Q4.

**Gaussian NLL** `gaussian_nll`:
\(\tfrac12\log(2\pi\sigma^2)+\tfrac{(y-\mu)^2}{2\sigma^2}\).

## Calibration

- `pit_gaussian` and `pit_sample` give the PIT values \(F_i(y_i)\). They are uniform when the model
  is calibrated. The sample version breaks ties at random.
- `calibration_curve` and `calibration_error`: observed \(P(\text{PIT}\le p)\) against \(p\), on
  levels 0.05–0.95. The trivial end points 0 and 1 are excluded. The function returns the mean and
  maximum absolute gap.
- `ence`: the expected normalized calibration error (Levi et al. 2022). Sort points by σ into
  equal-count bins, then compute the mean of \(|\mathrm{RMV}-\mathrm{RMSE}|/\mathrm{RMV}\). It
  compares **σ with the RMSE**; comparing σ with the MAE, as the v1.2 notebook's ECE did, is
  inconsistent (audit H3).

## Point

`rmse`, `mae`, `r2`, `bias`, and `point_report`, which returns all four.
