# Online conformal methods

The setting: at each time \(t\) we output an interval \(\hat C_t=[\hat\mu(x_t)-q_t,\ \hat\mu(x_t)+q_t]\),
then observe \(y_t\) and the score \(s_t=|y_t-\hat\mu(x_t)|\). Let
\(\mathrm{err}_t=\mathbb 1\{s_t>q_t\}\). No assumption is made about how the data are generated.

`OnlineConformalRegressor(estimator, method=…, alpha=…)` fits \(\hat\mu\) on the proper training part,
seeds the tracker with the calibration scores, and then runs on the test sequence:

```python
iv = ocr.predict_interval_online(X_test, y_test)   # row t uses only y_1..y_{t-1}
```

The tracker is **strictly causal**: `predict()` for step \(t\) is called before `update(s_t)`. The
test `test_online_methods_are_causal` perturbs \(y_t\) and checks that no interval at time \(\le t\)
changes. `update(X, y)` continues the sequence after the test period.

## ACI (`method="aci"`, class `ACI`)

Gibbs & Candès (2021). Track a level \(\alpha_t\) and use the empirical \((1-\alpha_t)\)-quantile of
past scores:

\[
\alpha_{t+1}=\alpha_t+\gamma(\alpha-\mathrm{err}_t),\qquad q_t=\mathrm{Quantile}_{1-\alpha_t}(s_1,\dots,s_{t-1}).
\]

If \(\alpha_t\le 0\), the interval is infinite; if \(\alpha_t\ge1\), it is empty. **Guarantee**,
for every sequence:
\(\big|\frac1T\sum_t\mathrm{err}_t-\alpha\big|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma T}\).
Parameters: `gamma` (default 0.005) and an optional `window` that restricts the score history.

## DtACI / FACI (`method="dtaci"`, classes `DtACI` = `FACI`)

Gibbs & Candès (2024). Run \(K\) ACI experts with step sizes \(\gamma_k\) and combine them with
exponential weights on the pinball loss of \(\alpha_t\), plus a share-back term \(\sigma\) that lets
weights recover after a regime change. The constants follow the reference implementation
(Bhatnagar et al. 2023, `online_conformal`):

\[
\gamma_k=0.001\cdot 2^{k},\ k=0..7;\quad I=100;\quad \sigma=\tfrac{1}{2I};\quad
\eta=\sqrt{\tfrac{3}{I}}\sqrt{\tfrac{\log(IK)+2}{\big((1-\alpha)^2\alpha^3+\alpha^2(1-\alpha)^3\big)/3}} .
\]

**Guarantee:** strongly adaptive regret, meaning low regret on every interval of length \(I\). You do
not have to pick γ.

## SF-OGD (`method="sf_ogd"`, class `ScaleFreeOGD`)

Bhatnagar et al. (2023). Online gradient descent on the pinball loss of \(q_t\) itself, with a
scale-free step:

\[
g_t=\alpha-\mathbb 1\{s_t>q_t\}\ (\text{sub-gradient of the pinball loss in } q),\qquad
q_{t+1}=\max\Big(0,\ q_t-\frac{D}{\sqrt{3\sum_{\tau\le t}g_\tau^2}}\,g_t\Big),
\]

with \(D=\sqrt3\cdot\max(\text{calibration score})\) unless `scale` is given. When the interval
misses, \(g_t<0\) and \(q\) grows.

## SAOCP (`method="saocp"`, class `SAOCP`)

Bhatnagar et al. (2023), Strongly Adaptive Online Conformal Prediction. Experts are SF-OGD
learners started at geometric times with lifetime `lifetime`. They are aggregated with
coin-betting weights, which have no learning rate to tune. This is a port of the Apache-2.0
reference code.

## Online NexCP (`method="nexcp"`, class `NexCPOnline`)

A weighted conformal quantile with geometric weights over the growing history of scores. See
[NexCP](nexcp.md).

!!! danger "v1.2 notebook errors"
    The old SF-OGD had the wrong gradient sign and updated with \(y_t\) before predicting \(t\). The old
    "FACI" picked its window using a loss that included \(y_t\). The old "SAOCP" was a fixed 500-point
    window, and ACI was never implemented (audit NX-4 to NX-8).
