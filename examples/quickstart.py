"""Minimal reguq example on synthetic heteroscedastic data (runs in seconds)."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from reguq.conformal import ConformalizedQuantileRegressor, SplitConformalRegressor
from reguq.metrics import interval_report

rng = np.random.default_rng(0)
X = rng.uniform(-2, 2, size=(3000, 3))
y = X @ [1.0, 2.0, 3.0] + (0.5 + np.abs(X[:, 0])) * rng.standard_normal(3000)
X_train, y_train, X_test, y_test = X[:2000], y[:2000], X[2000:], y[2000:]

gbr = GradientBoostingRegressor(random_state=0)
for name, model in {
    "split": SplitConformalRegressor(gbr, random_state=0),
    "normalized": SplitConformalRegressor(gbr, score="normalized", random_state=0),
    "cqr": ConformalizedQuantileRegressor(gbr, alpha=0.1, random_state=0),
}.items():
    iv = model.fit(X_train, y_train).predict_interval(X_test, 0.1)
    rep = interval_report(y_test, iv, alpha=0.1)
    print(f"{name:>10}: coverage={rep['coverage']:.3f}  width={rep['mean_width']:.2f}  "
          f"interval_score={rep['interval_score']:.2f}")
