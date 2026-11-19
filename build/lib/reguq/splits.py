"""Explicit train / calibration / test splitting.

Conformal guarantees require that conformity scores are computed on data the
model was **not** fitted on. These helpers make the calibration set a
first-class object instead of an ad-hoc split hidden inside each method.

``time_ordered=True`` never shuffles: the calibration block is taken from the
end of the training period, so it always precedes the test period. Use it for
time series (streamflow, sediment load, ...); shuffling a time series breaks
the ordering that sequential methods (NexCP, ACI, ...) rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._base import ArrayLike


def _n_samples(X: ArrayLike) -> int:
    return len(X)


def _take(X: ArrayLike, idx: np.ndarray) -> Any:
    """Index rows of numpy arrays or pandas objects positionally."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _resolve_size(size: float | int, n: int, name: str) -> int:
    if isinstance(size, int | np.integer) and not isinstance(size, bool):
        k = int(size)
    else:
        if not 0.0 < float(size) < 1.0:
            raise ValueError(f"{name} as a fraction must lie in (0, 1); got {size!r}")
        k = int(np.floor(float(size) * n))
    if not 1 <= k < n:
        raise ValueError(f"{name}={size!r} leaves an empty split for n={n}")
    return k


def split_indices(
    n: int,
    calib_size: float | int = 0.25,
    *,
    time_ordered: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_idx, calib_idx)`` for ``n`` samples.

    With ``time_ordered=True`` the calibration indices are the last
    ``calib_size`` positions, otherwise a seeded random subset.
    """
    k = _resolve_size(calib_size, n, "calib_size")
    if time_ordered:
        return np.arange(n - k), np.arange(n - k, n)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    return np.sort(perm[k:]), np.sort(perm[:k])


@dataclass
class TrainCalib:
    """Container returned by :func:`train_calib_split`."""

    X_train: Any
    X_calib: Any
    y_train: np.ndarray
    y_calib: np.ndarray
    train_idx: np.ndarray
    calib_idx: np.ndarray


def train_calib_split(
    X: ArrayLike,
    y: ArrayLike,
    calib_size: float | int = 0.25,
    *,
    time_ordered: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> TrainCalib:
    """Split ``(X, y)`` into a proper-training part and a calibration part."""
    n = _n_samples(X)
    if len(y) != n:
        raise ValueError(f"X has {n} rows but y has {len(y)}")
    tr, cal = split_indices(n, calib_size, time_ordered=time_ordered, random_state=random_state)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    return TrainCalib(_take(X, tr), _take(X, cal), y_arr[tr], y_arr[cal], tr, cal)


def kfold_indices(
    n: int,
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Partition ``range(n)`` into ``n_splits`` folds (list of index arrays).

    Used by CV+ / Jackknife+, where each point needs exactly one held-out
    model. ``shuffle=False`` gives contiguous blocks.
    """
    if not 2 <= n_splits <= n:
        raise ValueError(f"n_splits must be in [2, n={n}]; got {n_splits}")
    order = np.random.default_rng(random_state).permutation(n) if shuffle else np.arange(n)
    return [np.sort(f) for f in np.array_split(order, n_splits)]
