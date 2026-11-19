r"""Extreme-value (peaks-over-threshold) extrapolation of the conformal quantile.

For very small :math:`\alpha` (:math:`\alpha < 1/(n+1)`) the conformal
quantile is :math:`+\infty`. A generalized Pareto distribution (GPD) fitted
to calibration scores above a threshold :math:`u` extrapolates the tail
(Pickands 1975; Coles 2001, Sec. 4.3):

.. math::
    P(S > x) \approx \frac{N_u}{n}\Big(1 + \xi\,\frac{x-u}{\sigma}\Big)^{-1/\xi}
    \;\Rightarrow\;
    \hat q_{1-\alpha} = u + \frac{\sigma}{\xi}\Big[\Big(\frac{n\alpha}{N_u}\Big)^{-\xi} - 1\Big],

with :math:`\hat q = u + \sigma\log(N_u / (n\alpha))` when :math:`\xi = 0`.
:math:`N_u` is the number of exceedances among :math:`n` calibration scores.

This is a **semi-parametric estimate without a finite-sample coverage
guarantee**: it relies on the GPD tail approximation. When
:math:`\alpha \ge N_u/n` the requested quantile lies below the threshold and
the ordinary conformal quantile (which *is* guaranteed) is returned instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import genpareto

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, predict_1d
from ..splits import train_calib_split
from ._quantile import conformal_quantile


def gpd_tail_quantile(
    scores: ArrayLike, alpha: float, threshold: float = 0.9, min_exceedances: int = 10
) -> dict[str, float]:
    """Fit a GPD above the ``threshold`` empirical quantile and return the ``1-alpha`` quantile.

    Returns a dict with ``q``, ``u``, ``xi``, ``sigma``, ``n_exceed`` and
    ``method`` (``"gpd"`` or ``"conformal"``).
    """
    s = as_1d_float(scores, "scores")
    check_alpha(alpha)
    n = s.size
    u = float(np.quantile(s, threshold))
    exc = s[s > u] - u
    n_u = exc.size
    if n_u < min_exceedances:
        raise ValueError(
            f"only {n_u} exceedances above the {threshold} quantile; need {min_exceedances}. "
            "Lower `threshold` or use more calibration data."
        )
    if alpha >= n_u / n:
        return {
            "q": conformal_quantile(s, alpha),
            "u": u,
            "xi": np.nan,
            "sigma": np.nan,
            "n_exceed": n_u,
            "method": "conformal",
        }
    xi, _, sigma = genpareto.fit(exc, floc=0.0)
    ratio = n * alpha / n_u
    q = u - sigma * np.log(ratio) if abs(xi) < 1e-8 else u + sigma / xi * (ratio ** (-xi) - 1.0)
    return {
        "q": float(q),
        "u": u,
        "xi": float(xi),
        "sigma": float(sigma),
        "n_exceed": n_u,
        "method": "gpd",
    }


class ExtremeValueConformalRegressor(BaseIntervalRegressor):
    """Symmetric interval ``mu(x) ± q`` with a GPD-extrapolated radius for small ``alpha``.

    Coverage is two-sided: evaluate it with :func:`reguq.metrics.coverage`
    (``lower <= y <= upper``), not one-sided.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        threshold: float = 0.9,
        min_exceedances: int = 10,
        calib_size: float | int = 0.25,
        time_ordered: bool = False,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.threshold = threshold
        self.min_exceedances = min_exceedances
        self.calib_size = calib_size
        self.time_ordered = time_ordered
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_calib: ArrayLike | None = None,
        y_calib: ArrayLike | None = None,
    ) -> ExtremeValueConformalRegressor:
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
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_fitted(self, "calib_scores_")
        return predict_1d(self.estimator_, X)

    def tail_fit(self, alpha: float) -> dict[str, float]:
        """Diagnostics of the tail fit at ``alpha`` (see :func:`gpd_tail_quantile`)."""
        check_fitted(self, "calib_scores_")
        return gpd_tail_quantile(self.calib_scores_, alpha, self.threshold, self.min_exceedances)

    def predict_interval(self, X: ArrayLike, alpha: float = 0.01) -> np.ndarray:
        q = self.tail_fit(alpha)["q"]
        mu = self.predict(X)
        return as_interval(mu - q, mu + q)
