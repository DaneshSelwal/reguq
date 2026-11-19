r"""Resampling-based conformal methods: Jackknife+, CV+, Jackknife+-after-bootstrap.

Unlike split conformal, these use every training point both for fitting and
for calibration, at the price of fitting several models.

CV+ / Jackknife+ (Barber, Candès, Ramdas & Tibshirani 2021)
    Partition the data into K folds; :math:`\hat\mu_{-k}` is fitted without
    fold :math:`k`, and :math:`R_i = |Y_i - \hat\mu_{-k(i)}(X_i)|`. Then

    .. math::
        \hat C(x) = \big[\hat q^-_{\alpha}\{\hat\mu_{-k(i)}(x) - R_i\},\;
                         \hat q^+_{\alpha}\{\hat\mu_{-k(i)}(x) + R_i\}\big]

    where :math:`\hat q^-` is the :math:`\lfloor\alpha(n+1)\rfloor`-th smallest
    and :math:`\hat q^+` the :math:`\lceil(1-\alpha)(n+1)\rceil`-th smallest
    value. Guarantee: coverage :math:`\ge 1-2\alpha` (Jackknife+, K = n) and
    :math:`\ge 1-2\alpha-\sqrt{2/n}` (CV+); typically close to :math:`1-\alpha`.
    ``method="minmax"`` uses :math:`[\min_k\hat\mu_{-k}(x) - \hat q,\;
    \max_k\hat\mu_{-k}(x) + \hat q]` and guarantees :math:`\ge 1-\alpha`.

    With ``weights``, the quantiles become weighted quantiles with the test
    point's mass at :math:`\pm\infty` - the non-exchangeable jackknife+ /
    CV+ of Barber, Candès, Ramdas & Tibshirani (2023, Sec. 3).

Jackknife+-after-bootstrap (Kim, Xu & Barber 2020)
    Fit B bootstrap models once. For each i, :math:`\hat\mu_{-i}` aggregates
    the models whose bootstrap sample did **not** contain i (out-of-bag),
    :math:`R_i = |Y_i - \hat\mu_{-i}(X_i)|`, and the interval is built exactly
    as in Jackknife+ from :math:`\hat\mu_{-i}(x) \pm R_i`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, n_rows, predict_1d, take_rows
from ..splits import kfold_indices
from ._quantile import conformal_quantile, weighted_conformal_quantile, weighted_lower_quantile


def _plus_interval(
    centers: np.ndarray, residuals: np.ndarray, alpha: float, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife+ bounds. ``centers`` has shape (n_calib, n_test)."""
    low_vals = centers - residuals[:, None]
    high_vals = centers + residuals[:, None]
    n = residuals.size
    if weights is None:
        k_lo = int(math.floor(alpha * (n + 1) + 1e-10))
        k_hi = int(math.ceil((1 - alpha) * (n + 1) - 1e-10))
        lo = (
            np.full(centers.shape[1], -np.inf)
            if k_lo < 1
            else np.partition(low_vals, k_lo - 1, axis=0)[k_lo - 1]
        )
        hi = (
            np.full(centers.shape[1], np.inf)
            if k_hi > n
            else np.partition(high_vals, k_hi - 1, axis=0)[k_hi - 1]
        )
        return lo, hi
    lo = np.array(
        [weighted_lower_quantile(low_vals[:, j], weights, alpha) for j in range(centers.shape[1])]
    )
    hi = np.array(
        [
            weighted_conformal_quantile(high_vals[:, j], weights, alpha)
            for j in range(centers.shape[1])
        ]
    )
    return lo, hi


def _minmax_interval(
    centers: np.ndarray, residuals: np.ndarray, alpha: float, weights: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    q = (
        conformal_quantile(residuals, alpha)
        if weights is None
        else weighted_conformal_quantile(residuals, weights, alpha)
    )
    return centers.min(axis=0) - q, centers.max(axis=0) + q


def _check_weights(weights: ArrayLike | None, n: int) -> np.ndarray | None:
    if weights is None:
        return None
    w = as_1d_float(weights, "weights")
    if w.size != n:
        raise ValueError(f"weights must have one entry per training point ({n}); got {w.size}")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    return w


class CrossConformalRegressor(BaseIntervalRegressor):
    """CV+ / Jackknife+ / CV-minmax prediction intervals.

    Parameters
    ----------
    estimator
        Scikit-learn style regressor.
    n_splits
        Number of folds K; ``"loo"`` gives the Jackknife+ (K = n).
    method
        ``"plus"`` (default) or ``"minmax"``.
    shuffle
        Shuffle before forming folds. Use ``False`` for time series.
    weights
        Optional per-training-point weights (in data order) for the
        non-exchangeable variant; see :func:`reguq.conformal.exponential_weights`.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        n_splits: int | str = 5,
        method: str = "plus",
        shuffle: bool = True,
        weights: ArrayLike | None = None,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.n_splits = n_splits
        self.method = method
        self.shuffle = shuffle
        self.weights = weights
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> CrossConformalRegressor:
        if self.method not in ("plus", "minmax"):
            raise ValueError("method must be 'plus' or 'minmax'")
        y_arr = as_1d_float(y)
        n = n_rows(X)
        k = n if self.n_splits == "loo" else int(self.n_splits)
        self.weights_ = _check_weights(self.weights, n)
        folds = kfold_indices(n, k, shuffle=self.shuffle and k < n, random_state=self.random_state)
        self.fold_of_ = np.empty(n, dtype=int)
        self.models_ = []
        residuals = np.empty(n)
        all_idx = np.arange(n)
        for f, held in enumerate(folds):
            train = np.setdiff1d(all_idx, held, assume_unique=True)
            m = fit_clone(self.estimator, take_rows(X, train), y_arr[train])
            self.models_.append(m)
            self.fold_of_[held] = f
            residuals[held] = np.abs(y_arr[held] - predict_1d(m, take_rows(X, held)))
        self.residuals_ = residuals
        self.estimator_ = fit_clone(self.estimator, X, y_arr)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Prediction of the model refitted on all training data."""
        check_fitted(self, "residuals_")
        return predict_1d(self.estimator_, X)

    def predict_interval(
        self, X: ArrayLike, alpha: float = 0.1, *, batch_size: int = 2048
    ) -> np.ndarray:
        check_fitted(self, "residuals_")
        a = check_alpha(alpha)
        fold_preds = np.stack([predict_1d(m, X) for m in self.models_])  # (K, n_test)
        out_lo, out_hi = [], []
        for start in range(0, fold_preds.shape[1], batch_size):
            centers = fold_preds[self.fold_of_, start : start + batch_size]  # (n, b)
            fn = _plus_interval if self.method == "plus" else _minmax_interval
            lo, hi = fn(centers, self.residuals_, a, self.weights_)
            out_lo.append(lo)
            out_hi.append(hi)
        return as_interval(np.concatenate(out_lo), np.concatenate(out_hi))


class JackknifePlusAfterBootstrapRegressor(BaseIntervalRegressor):
    """Jackknife+-after-bootstrap, J+aB (Kim, Xu & Barber 2020).

    Parameters
    ----------
    estimator
        Scikit-learn style regressor.
    n_bootstraps
        Number of bootstrap models B.
    aggregation
        ``"mean"`` or ``"median"`` of out-of-bag model predictions.
    method
        ``"plus"`` or ``"minmax"``.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        n_bootstraps: int = 30,
        aggregation: str = "mean",
        method: str = "plus",
        weights: ArrayLike | None = None,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.n_bootstraps = n_bootstraps
        self.aggregation = aggregation
        self.method = method
        self.weights = weights
        self.random_state = random_state

    def _aggregate(self, preds: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Aggregate rows of ``preds`` (B, m) selected per point by boolean ``mask`` (n, B) -> (n, m)."""
        if self.aggregation == "mean":
            m = mask.astype(float)
            return (m @ preds) / m.sum(axis=1, keepdims=True)
        return np.stack([np.median(preds[row], axis=0) for row in mask])

    def fit(self, X: ArrayLike, y: ArrayLike) -> JackknifePlusAfterBootstrapRegressor:
        if self.aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be 'mean' or 'median'")
        if self.method not in ("plus", "minmax"):
            raise ValueError("method must be 'plus' or 'minmax'")
        y_arr = as_1d_float(y)
        n = n_rows(X)
        rng = np.random.default_rng(self.random_state)
        self.models_ = []
        in_bag = np.zeros((n, self.n_bootstraps), dtype=bool)
        for b in range(self.n_bootstraps):
            idx = rng.integers(0, n, size=n)
            in_bag[idx, b] = True
            self.models_.append(fit_clone(self.estimator, take_rows(X, idx), y_arr[idx]))
        oob = ~in_bag
        keep = oob.any(axis=1)  # points with at least one out-of-bag model
        if not keep.any():
            raise ValueError("no point is out-of-bag for any model; increase n_bootstraps")
        self.oob_mask_ = oob[keep]
        self.n_dropped_ = int((~keep).sum())
        kept = np.flatnonzero(keep)
        train_preds = np.stack(
            [predict_1d(m, take_rows(X, kept)) for m in self.models_]
        )  # (B, n_kept)
        # mu_{-i}(X_i): aggregate over the models for which i is out-of-bag.
        if self.aggregation == "mean":
            loo_at_own = (self.oob_mask_ * train_preds.T).sum(axis=1) / self.oob_mask_.sum(axis=1)
        else:
            loo_at_own = np.array(
                [np.median(train_preds[row, i]) for i, row in enumerate(self.oob_mask_)]
            )
        self.residuals_ = np.abs(y_arr[keep] - loo_at_own)
        w = _check_weights(self.weights, n)
        self.weights_ = None if w is None else w[keep]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Bagged prediction (aggregate of all bootstrap models)."""
        check_fitted(self, "residuals_")
        preds = np.stack([predict_1d(m, X) for m in self.models_])
        return preds.mean(axis=0) if self.aggregation == "mean" else np.median(preds, axis=0)

    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray:
        check_fitted(self, "residuals_")
        a = check_alpha(alpha)
        preds = np.stack([predict_1d(m, X) for m in self.models_])  # (B, m)
        centers = self._aggregate(preds, self.oob_mask_)
        fn = _plus_interval if self.method == "plus" else _minmax_interval
        lo, hi = fn(centers, self.residuals_, a, self.weights_)
        return as_interval(lo, hi)
