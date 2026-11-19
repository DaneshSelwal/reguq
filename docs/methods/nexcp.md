# Non-exchangeable conformal prediction (NexCP)

Barber, Candès, Ramdas & Tibshirani (2023), *Conformal prediction beyond exchangeability*.

## Method: `NexCPRegressor`

Fit \(\hat\mu\) on the proper training part and compute calibration scores
\(S_i=|Y_i-\hat\mu(X_i)|\) in time order. Fix weights \(w_i\in[0,1]\) **in advance**, without looking
at the data. reguq's default is geometric recency weighting,

\[
w_i=\rho^{\,n+1-i},\qquad i=1,\dots,n,\quad w_{n+1}=1
\]

(`exponential_weights(n, rho)`), so the most recent score gets weight ρ and a score \(k\) steps
older gets \(\rho^{k+1}\). The interval is \(\hat\mu(x)\pm\hat q\), where \(\hat q\) is the
[weighted conformal quantile](conformal-foundations.md#weighted-quantile) **including the
\(+\infty\) mass of the test point**.

## Guarantee

For **any** joint distribution (Barber et al. 2023, Thm 2a, for a fixed model):

\[
P\big(Y_{n+1}\in\hat C(X_{n+1})\big)\ \ge\ 1-\alpha-\sum_{i=1}^{n}\tilde w_i\, d_{\mathrm{TV}}\big(R(Z),R(Z^{i})\big),
\]

where \(Z^{i}\) swaps data point \(i\) with the test point and \(\tilde w_i=w_i/(1+\sum_j w_j)\). If
the data are exchangeable, the drift term is zero. If the distribution drifts slowly, putting little
weight on old points keeps the term small.

**Choosing ρ:** the effective sample size is about \(1/(1-\rho)\). For example, ρ = 0.99 gives about
100 recent points. A smaller ρ adapts faster but makes \(\hat q\) noisier. With \(\alpha=0.1\), you
need roughly \(1/(1-\rho)\gtrsim 10\) for \(\hat q\) to be finite.

`weights=` accepts any fixed weight vector instead of the geometric default.

## Online variant

`OnlineConformalRegressor(method="nexcp")` uses `NexCPOnline`. Its score history grows with each
observed outcome, and the weights are recomputed every step. See [online methods](online.md).

!!! danger "v1.2 notebook errors"
    The notebooks computed weighted quantiles of **in-sample training residuals** (no calibration
    set), normalized the weights without the test point's \(+\infty\) mass, and interpolated
    linearly (audit NX-1, NX-2).
