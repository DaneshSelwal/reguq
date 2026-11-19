"""Small internal helpers shared by estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone

from ._base import ArrayLike


def take_rows(X: ArrayLike, idx: np.ndarray) -> Any:
    """Positional row selection for numpy arrays and pandas objects."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def n_rows(X: ArrayLike) -> int:
    return len(X)


def fit_clone(estimator: Any, X: ArrayLike, y: np.ndarray, **fit_params: Any) -> Any:
    """Fit an unfitted copy of ``estimator`` (never mutates the user's object)."""
    est = clone(estimator)
    est.fit(X, y, **fit_params)
    return est


def predict_1d(estimator: Any, X: ArrayLike) -> np.ndarray:
    """``estimator.predict(X)`` as a 1-D float array."""
    pred = np.asarray(estimator.predict(X), dtype=float)
    if pred.ndim == 2 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim != 1:
        raise ValueError(f"estimator.predict returned shape {pred.shape}; expected (n,)")
    return pred


def check_fitted(obj: Any, attr: str) -> None:
    if not hasattr(obj, attr):
        raise RuntimeError(f"{type(obj).__name__} is not fitted yet; call fit() first")
