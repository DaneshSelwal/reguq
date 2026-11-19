# Quick start

All interval estimators share one contract:

| Method | Returns |
|---|---|
| `fit(X, y)` | `self`; holds out its own calibration data from `(X, y)` |
| `predict(X)` | point predictions, shape `(n,)` |
| `predict_interval(X, alpha)` | `[lower, upper]`, shape `(n, 2)`, target coverage `1 - alpha` |
| `predict_dist(X)` | distributional models only: a `GaussianPrediction` or `SamplePrediction` |

They are scikit-learn estimators, so `clone`, `get_params` and pipelines work.

## A first interval

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from reguq.conformal import SplitConformalRegressor
from reguq.metrics import interval_report

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, (3000, 3))
y = X @ [1.0, 2.0, 3.0] + (0.5 + abs(X[:, 0])) * rng.standard_normal(3000)
X_train, y_train, X_test, y_test = X[:2000], y[:2000], X[2000:], y[2000:]

cp = SplitConformalRegressor(GradientBoostingRegressor(), calib_size=0.25, random_state=0)
cp.fit(X_train, y_train)
iv = cp.predict_interval(X_test, alpha=0.1)
print(interval_report(y_test, iv, alpha=0.1))
```

`interval_report` returns coverage, mean width, normalized width, the interval (Winkler) score, CWC
and the worst-bin coverage.

## Adaptive width

The noise above grows with \(|x_1|\). An absolute score gives every point the same width, so it
over-covers easy points and under-covers hard ones. Two ways to adapt the width:

```python
from reguq.conformal import ConformalizedQuantileRegressor

normalized = SplitConformalRegressor(GradientBoostingRegressor(), score="normalized", random_state=0)
cqr = ConformalizedQuantileRegressor(GradientBoostingRegressor(), alpha=0.1, random_state=0)
```

Both keep the finite-sample guarantee. Compare their `interval_score` (lower is better) and
`ssc` (worst-bin coverage, closer to 0.9 is better).

## A predictive distribution

```python
from reguq.probabilistic import NGBoostRegressor

ngb = NGBoostRegressor(n_estimators=500).fit(X_train, y_train)
dist = ngb.predict_dist(X_test)
dist.crps(y_test), dist.nll(y_test), dist.interval(0.1)
```

A model-based interval has no coverage guarantee. To add one, fit the model on part of the training
data and use its predicted standard deviation as the scale of a normalized conformal score on the
remaining part (`prefit=True` means the estimators are already fitted):

```python
from reguq.probabilistic import StdEstimator
from reguq.splits import train_calib_split

s = train_calib_split(X_train, y_train, calib_size=0.25, random_state=0)
ngb = NGBoostRegressor(n_estimators=500).fit(s.X_train, s.y_train)
cp = SplitConformalRegressor(
    ngb, score="normalized", scale_estimator=StdEstimator(ngb), prefit=True
).fit(s.X_calib, s.y_calib)
cp.predict_interval(X_test, alpha=0.1)   # >= 90 % marginal coverage
```

## Next

- [Choosing a method](choosing.md)
- [Time series and drift](time-series.md)
- [Pipeline and CLI](pipeline.md) to run many models × methods on CSV files
