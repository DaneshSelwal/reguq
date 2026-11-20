r"""Multi-quantile regression with non-crossing guarantees.

Separately fitted quantile models can *cross* (e.g. :math:`\hat q_{0.2} >
\hat q_{0.3}`), which yields negative interval widths and invalid CRPS. The
monotone rearrangement of Chernozhukov, Fernández-Val & Galichon (2010) -
sorting each row of predicted quantiles - removes crossing and never
increases the pinball loss averaged over the grid.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, predict_1d
from ..metrics import crps_quantile
from ._factory import with_quantile

DEFAULT_QUANTILES = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))


def rearrange_quantiles(q: ArrayLike) -> np.ndarray:
    """Sort each row of an ``(n, K)`` quantile matrix (monotone rearrangement)."""
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expected an (n, K) array of quantile predictions")
    return np.sort(arr, axis=1)


def crossing_rate(q: ArrayLike) -> float:
    """Fraction of rows with at least one crossing (``q[:, k] > q[:, k+1]``)."""
    arr = np.asarray(q, dtype=float)
    return float((np.diff(arr, axis=1) < 0).any(axis=1).mean())


class MultiQuantileRegressor(BaseIntervalRegressor):
    """One quantile model per level, with optional rearrangement.

    Intervals from this class are **not calibrated**: for finite-sample
    coverage wrap the estimator with
    :class:`reguq.conformal.ConformalizedQuantileRegressor`.

    Parameters
    ----------
    estimator
        A regressor :func:`with_quantile` can configure, used for every level.
    quantiles
        Strictly increasing levels in (0, 1). Requesting an interval at
        ``alpha`` needs ``alpha/2`` and ``1 - alpha/2`` in the grid, or uses
        linear interpolation between neighbouring fitted levels.
    rearrange
        Sort predicted quantiles per row (recommended).
    """

    def __init__(
        self,
        estimator: Any,
        *,
        quantiles: ArrayLike = DEFAULT_QUANTILES,
        rearrange: bool = True,
    ) -> None:
        self.estimator = estimator
        self.quantiles = quantiles
        self.rearrange = rearrange

    def fit(self, X: ArrayLike, y: ArrayLike) -> MultiQuantileRegressor:
        taus = np.asarray(self.quantiles, dtype=float)
        if (np.diff(taus) <= 0).any() or taus.min() <= 0 or taus.max() >= 1:
            raise ValueError("quantiles must be strictly increasing inside (0, 1)")
        y_arr = as_1d_float(y)
        self.quantiles_ = taus
        self.models_ = [fit_clone(with_quantile(self.estimator, float(t)), X, y_arr) for t in taus]
        return self

    def predict_quantiles(self, X: ArrayLike, *, rearrange: bool | None = None) -> np.ndarray:
        """Predicted quantiles, shape ``(n, K)`` in the order of ``quantiles_``."""
        check_fitted(self, "models_")
        q = np.column_stack([predict_1d(m, X) for m in self.models_])
        do = self.rearrange if rearrange is None else rearrange
        return rearrange_quantiles(q) if do else q

    def _at(self, q: np.ndarray, tau: float) -> np.ndarray:
        taus = self.quantiles_
        if tau < taus[0] or tau > taus[-1]:
            raise ValueError(f"level {tau} is outside the fitted grid [{taus[0]}, {taus[-1]}]")
        j = int(np.searchsorted(taus, tau))
        if j < taus.size and np.isclose(taus[j], tau):
            return q[:, j]
        w = (tau - taus[j - 1]) / (taus[j] - taus[j - 1])
        return (1 - w) * q[:, j - 1] + w * q[:, j]

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Median prediction (interpolated if 0.5 is not in the grid)."""
        return self._at(self.predict_quantiles(X), 0.5)

    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray:
        a = check_alpha(alpha)
        q = self.predict_quantiles(X)
        return as_interval(self._at(q, a / 2), self._at(q, 1 - a / 2))

    def crps(self, X: ArrayLike, y: ArrayLike) -> float:
        """Quantile-grid CRPS approximation with proper level weights."""
        return float(crps_quantile(y, self.predict_quantiles(X), self.quantiles_))


class QuantileForestRegressor(BaseIntervalRegressor):
    """Quantile regression forest (Meinshausen 2006) via ``quantile-forest``.

    Unlike quantiles of per-tree *means* (which only measure tree
    disagreement and ignore the noise in y), a QRF estimates the conditional
    distribution of y from the leaf co-membership weights.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf: int = 5,
        random_state: int | None = None,
        forest_params: dict[str, Any] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.forest_params = forest_params

    def fit(self, X: ArrayLike, y: ArrayLike) -> QuantileForestRegressor:
        from ..probabilistic._base import require

        qf = require("quantile_forest", "qrf")
        self.model_ = qf.RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            **(self.forest_params or {}),
        )
        self.model_.fit(X, as_1d_float(y))
        return self

    def predict_quantiles(self, X: ArrayLike, quantiles: ArrayLike) -> np.ndarray:
        check_fitted(self, "model_")
        q = np.asarray(
            self.model_.predict(X, quantiles=list(np.atleast_1d(quantiles))), dtype=float
        )
        return q.reshape(q.shape[0], -1)

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.predict_quantiles(X, [0.5])[:, 0]

    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray:
        a = check_alpha(alpha)
        q = self.predict_quantiles(X, [a / 2, 1 - a / 2])
        return as_interval(q[:, 0], q[:, 1])
