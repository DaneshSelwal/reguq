"""Uniform test-set evaluation of any reguq estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..metrics import calibration_error, interval_report, point_report


def evaluate(
    est: Any, X_test: Any, y_test: np.ndarray, alpha: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Point, interval and (if available) distributional metrics on the test set."""
    online = hasattr(est, "predict_interval_online")
    iv = (
        est.predict_interval_online(X_test, y_test)
        if online
        else est.predict_interval(X_test, alpha)
    )
    pred = est.predict(X_test)
    row: dict[str, Any] = {**point_report(y_test, pred), **interval_report(y_test, iv, alpha)}
    if hasattr(est, "predict_dist"):
        d = est.predict_dist(X_test)
        row["crps"] = d.crps(y_test)
        row["nll"] = d.nll(y_test)
        row.update({f"pit_{k}": v for k, v in calibration_error(d.pit(y_test)).items()})
    if hasattr(est, "alpha_tilde_"):
        row["alpha_tilde"] = est.alpha_tilde_
    preds = pd.DataFrame({"y_true": y_test, "y_pred": pred, "lower": iv[:, 0], "upper": iv[:, 1]})
    return row, preds
