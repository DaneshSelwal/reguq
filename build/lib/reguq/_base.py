"""Shared estimator contract and input validation.

Every interval-producing estimator in reguq follows one contract:

* ``fit(X, y)`` returns ``self``;
* ``predict(X)`` returns the point prediction, shape ``(n,)``;
* ``predict_interval(X, alpha)`` returns an array of shape ``(n, 2)`` whose
  columns are the lower and upper bound of a ``1 - alpha`` prediction
  interval.

Distributional estimators additionally expose ``predict_dist(X)`` (see
:mod:`reguq.probabilistic`). Keeping one contract means every method can be
evaluated with the same metrics and compared like-for-like.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

ArrayLike = Any


def check_alpha(alpha: float) -> float:
    """Validate a miscoverage level ``alpha`` in the open interval (0, 1)."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    return alpha


def as_1d_float(y: ArrayLike, name: str = "y") -> np.ndarray:
    """Return ``y`` as a contiguous 1-D float array (accepts pandas objects)."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def as_interval(lower: ArrayLike, upper: ArrayLike) -> np.ndarray:
    """Stack lower/upper bounds into an ``(n, 2)`` array, validating shapes."""
    lo = as_1d_float(lower, "lower")
    hi = as_1d_float(upper, "upper")
    if lo.shape != hi.shape:
        raise ValueError(f"lower {lo.shape} and upper {hi.shape} differ in shape")
    return np.column_stack([lo, hi])


def check_interval(interval: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split an ``(n, 2)`` interval array into (lower, upper) 1-D arrays."""
    arr = np.asarray(interval, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"interval must have shape (n, 2); got {arr.shape}")
    return arr[:, 0], arr[:, 1]


class BaseIntervalRegressor(RegressorMixin, BaseEstimator):
    """Base class for estimators that output prediction intervals.

    Subclasses implement :meth:`fit`, :meth:`predict` and
    :meth:`predict_interval`. ``score`` (from :class:`RegressorMixin`) works
    because :meth:`predict` always returns a plain 1-D array.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike) -> BaseIntervalRegressor: ...

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray: ...

    @abstractmethod
    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray: ...
