r"""Non-exchangeable split conformal prediction, NexCP (Barber et al. 2023).

Calibration scores are weighted by fixed (data-independent) weights, typically
decaying with age so that recent observations matter more under drift:

.. math:: w_i = \rho^{\,n+1-i}, \qquad i = 1,\dots,n \;(\text{oldest} \to \text{newest}).

The interval uses :func:`~reguq.conformal.weighted_conformal_quantile`, which
includes the test point's mass at :math:`+\infty`. Barber et al. (2023,
Thm. 2) show coverage :math:`\ge 1-\alpha-\sum_i \tilde w_i\, d_{TV}(R, R^i)`:
exact when the data are exchangeable, and degrading gracefully with the
amount of distribution drift.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, predict_1d
from ..splits import train_calib_split
from ._quantile import weighted_conformal_quantile


def exponential_weights(n: int, rho: float = 0.99) -> np.ndarray:
    """Weights ``rho**(n+1-i)`` for ``i = 1..n`` (index 0 = oldest point)."""
    if not 0.0 < rho <= 1.0:
        raise ValueError("rho must lie in (0, 1]")
    return rho ** (n - np.arange(n, dtype=float))


class NexCPRegressor(BaseIntervalRegressor):
    """Split NexCP with exponentially decaying (or user-given) calibration weights.

    The calibration block is the most recent ``calib_size`` part of the
    training data (``time_ordered=True`` by default). For sequential use with
    the calibration set growing as test labels arrive, use
    :class:`reguq.conformal.OnlineConformalRegressor` with ``method="nexcp"``.

    Parameters
    ----------
    estimator
        Scikit-learn style regressor.
    rho
        Decay of :func:`exponential_weights`; ``rho=1`` is ordinary split CP.
    weights
        Optional explicit calibration weights (oldest first); overrides ``rho``.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        rho: float = 0.99,
        weights: ArrayLike | None = None,
        calib_size: float | int = 0.25,
        time_ordered: bool = True,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.rho = rho
        self.weights = weights
        self.calib_size = calib_size
        self.time_ordered = time_ordered
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_calib: ArrayLike | None = None,
        y_calib: ArrayLike | None = None,
    ) -> NexCPRegressor:
        if X_calib is None:
            s = train_calib_split(
                X,
                y,
                self.calib_size,
                time_ordered=self.time_ordered,
                random_state=self.random_state,
            )
            X_tr, y_tr, X_cal, y_cal = s.X_train, s.y_train, s.X_calib, s.y_calib
        else:
            X_tr, y_tr, X_cal, y_cal = X, as_1d_float(y), X_calib, as_1d_float(y_calib)
        self.estimator_ = fit_clone(self.estimator, X_tr, y_tr)
        self.calib_scores_ = np.abs(y_cal - predict_1d(self.estimator_, X_cal))
        n = self.calib_scores_.size
        if self.weights is not None:
            w = as_1d_float(self.weights, "weights")
            if w.size != n:
                raise ValueError(f"weights must have length n_calib={n}")
            self.calib_weights_ = w
        else:
            self.calib_weights_ = exponential_weights(n, self.rho)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_fitted(self, "calib_scores_")
        return predict_1d(self.estimator_, X)

    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray:
        check_fitted(self, "calib_scores_")
        q = weighted_conformal_quantile(self.calib_scores_, self.calib_weights_, check_alpha(alpha))
        mu = self.predict(X)
        return as_interval(mu - q, mu + q)
