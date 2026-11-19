# Jackknife+, CV+ and Jackknife+-after-bootstrap

Split conformal wastes data: the calibration part is not used for fitting. Resampling methods
use every point for both fitting and calibration, at the cost of \(K\) or \(B\) refits.

## CV+ and Jackknife+: `CrossConformalRegressor`

Barber, Candès, Ramdas & Tibshirani (2021). Split the data into \(K\) folds and fit
\(\hat\mu_{-k}\) without fold \(k\). For a point \(i\) in fold \(k(i)\), let
\(R_i=|Y_i-\hat\mu_{-k(i)}(X_i)|\). Then

\[
\hat C^{+}(x)=\Big[\hat q^{-}_{\alpha}\{\hat\mu_{-k(i)}(x)-R_i\},\ \hat q^{+}_{\alpha}\{\hat\mu_{-k(i)}(x)+R_i\}\Big],
\]

where \(\hat q^{+}_\alpha\) is the \(\lceil(1-\alpha)(n+1)\rceil\)-th smallest value and
\(\hat q^{-}_\alpha\) the \(\lfloor\alpha(n+1)\rfloor\)-th smallest. `n_splits="loo"` gives the
Jackknife+ (\(K=n\)).

| `method` | Interval | Guarantee |
|---|---|---|
| `"plus"` | above | Jackknife+ ≥ 1−2α; CV+ ≥ 1−2α − \(\min\{\tfrac{2(1-1/K)}{n/K+1},\tfrac{1-K/n}{K+1}\}\) |
| `"minmax"` | \([\min_k\hat\mu_{-k}(x)-\hat q,\ \max_k\hat\mu_{-k}(x)+\hat q]\) | ≥ 1−α |

In practice the "plus" intervals cover close to 1−α; the 1−2α bound is the worst case.

`shuffle=False` keeps the folds contiguous for time-ordered data. The pipeline sets this
automatically when `time_ordered=True`.

## Jackknife+-after-bootstrap: `JackknifePlusAfterBootstrapRegressor`

Kim, Xu & Barber (2020). Fit \(B\) bootstrap models. For each point \(i\), aggregate only the models
whose bootstrap sample **did not** contain \(i\) (out-of-bag):
\(\hat\mu_{-i}=\mathrm{agg}\{\hat\mu_b: i\notin \text{bag}_b\}\). Then form the Jackknife+ interval
above with \(R_i=|Y_i-\hat\mu_{-i}(X_i)|\). This costs \(B\) fits instead of \(n\). The guarantee
≥ 1−2α holds when \(B\) is drawn from a binomial distribution as in the paper; a fixed \(B\) is the
usual practical approximation.

`aggregation` is `"mean"` or `"median"`. A point that is in every bag has no out-of-bag models; this
is unlikely for \(B\ge 20\) and raises an error rather than being skipped silently.

!!! danger "v1.2 notebook errors"
    The old "J+aB" used **in-bag** residuals, and the old "CV+" averaged each fold's interval end
    points. Neither is the named method (audit NX-3). The MAPIE-based "J+aB" was actually J-minmax-aB
    (audit MP-3).
