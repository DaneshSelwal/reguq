"""Point-prediction metrics."""

from __future__ import annotations

import numpy as np

from .._base import ArrayLike, as_1d_float


def _pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y, p = as_1d_float(y_true), as_1d_float(y_pred, "y_pred")
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    return y, p


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, p = _pair(y_true, y_pred)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("y_true is constant; R^2 undefined")
    return float(1.0 - np.sum((y - p) ** 2) / ss_tot)


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean error ``mean(y_pred - y_true)``."""
    y, p = _pair(y_true, y_pred)
    return float(np.mean(p - y))
