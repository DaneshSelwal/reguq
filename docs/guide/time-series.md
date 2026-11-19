# Time series and drift

Conformal guarantees rest on **exchangeability**. Daily hydrological or sediment series are not
exchangeable: neighbouring days are correlated and the distribution drifts. reguq handles this in
three ways.

## 1. Never shuffle

Pass `time_ordered=True`. The calibration block is then the **last** `calib_size` share of the
training period, every resampling method uses ordered folds, and tuning uses forward-chaining CV
(`TimeSeriesSplit`).

```python
SplitConformalRegressor(model, calib_size=0.25, time_ordered=True)
```

Split conformal then calibrates on the most recent past. Under mild drift it is the best static
choice, but its guarantee no longer formally holds.

## 2. Weight recent data (NexCP)

`NexCPRegressor` gives calibration score \(i\) weight \(w_i=\rho^{\,n+1-i}\), so recent scores count
more. Its coverage loss is bounded by the weighted total-variation drift (Barber et al. 2023), so
with ρ < 1 old regimes stop mattering. See [NexCP](../methods/nexcp.md).

## 3. Adapt online

If the true value \(y_t\) becomes known after each prediction, update the interval sequentially:

```python
from reguq.conformal import OnlineConformalRegressor

aci = OnlineConformalRegressor(model, method="aci", alpha=0.1, method_params={"gamma": 0.005})
aci.fit(X_train, y_train)
iv = aci.predict_interval_online(X_test, y_test)   # row t only uses y_1..y_{t-1}
aci.alpha_history_, aci.radius_history_
```

ACI guarantees that the long-run miss rate approaches α **for any sequence**, including adversarial
ones. Its bound is

\[
\Big|\frac1T\sum_{t=1}^T \mathrm{err}_t-\alpha\Big|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma T}.
\]

DtACI runs several ACI experts with different step sizes γ and weights them, so you do not have to
choose γ. See [online methods](../methods/online.md).

!!! warning "Evaluating online methods"
    `predict_interval_online` needs `y_test` because each interval is updated with the outcomes
    **before** it. The test `test_online_methods_are_causal` checks that changing \(y_t\) never changes the
    interval at time \(t\) or earlier.
