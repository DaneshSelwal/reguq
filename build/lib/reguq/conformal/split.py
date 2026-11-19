r"""Split (inductive) conformal regression and conformalized quantile regression.

Both methods fit models on a *proper training* set and compute conformity
scores on a disjoint *calibration* set, which gives the finite-sample
marginal guarantee :math:`P(Y_{n+1} \in \hat C(X_{n+1})) \ge 1-\alpha` under
exchangeability (Lei et al. 2018; Romano, Patterson & Candès 2019).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, predict_1d
from ..quantile._factory import with_quantile
from ..splits import train_calib_split
from ._quantile import conformal_quantile


def _resolve_split(
    X: ArrayLike,
    y: ArrayLike,
    X_calib: ArrayLike | None,
    y_calib: ArrayLike | None,
    calib_size: float | int,
    time_ordered: bool,
    random_state: Any,
) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    if (X_calib is None) != (y_calib is None):
        raise ValueError("pass both X_calib and y_calib, or neither")
    if X_calib is not None:
        return X, as_1d_float(y), X_calib, as_1d_float(y_calib, "y_calib")
    s = train_calib_split(X, y, calib_size, time_ordered=time_ordered, random_state=random_state)
    return s.X_train, s.y_train, s.X_calib, s.y_calib


class SplitConformalRegressor(BaseIntervalRegressor):
    r"""Split conformal prediction with absolute or normalized residual scores.

    Parameters
    ----------
    estimator
        Any scikit-learn style regressor (cloned, never mutated).
    score
        ``"absolute"``: :math:`s = |y - \hat\mu(x)|`, interval
        :math:`\hat\mu(x) \pm \hat q`.
        ``"normalized"``: :math:`s = |y - \hat\mu(x)| / \hat\sigma(x)`, interval
        :math:`\hat\mu(x) \pm \hat q\,\hat\sigma(x)` (locally weighted CP,
        Lei et al. 2018 Sec. 5.2). :math:`\hat\sigma` is fitted to
        out-of-fold absolute residuals of the training part, so it is not
        biased low by in-sample overfitting; validity does not depend on it.
    scale_estimator
        Regressor for :math:`\hat\sigma` (default: a clone of ``estimator``).
    calib_size
        Fraction or count of training rows held out for calibration (ignored
        when ``X_calib``/``y_calib`` are passed to :meth:`fit`).
    time_ordered
        If True the calibration block is the *last* part of the data and
        nothing is shuffled.
    prefit
        If True, ``estimator`` is already fitted and :meth:`fit` only
        calibrates (all data passed to ``fit`` is used as calibration).
    random_state
        Seed for the random split.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        score: str = "absolute",
        scale_estimator: Any = None,
        calib_size: float | int = 0.25,
        time_ordered: bool = False,
        prefit: bool = False,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.score = score
        self.scale_estimator = scale_estimator
        self.calib_size = calib_size
        self.time_ordered = time_ordered
        self.prefit = prefit
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_calib: ArrayLike | None = None,
        y_calib: ArrayLike | None = None,
    ) -> SplitConformalRegressor:
        if self.score not in ("absolute", "normalized"):
            raise ValueError("score must be 'absolute' or 'normalized'")
        if self.prefit:
            if self.score == "normalized" and self.scale_estimator is None:
                raise ValueError("prefit with score='normalized' needs a fitted scale_estimator")
            self.estimator_ = self.estimator
            self.scale_estimator_ = self.scale_estimator
            X_cal, y_cal = (
                (X, as_1d_float(y)) if X_calib is None else (X_calib, as_1d_float(y_calib))
            )
        else:
            X_tr, y_tr, X_cal, y_cal = _resolve_split(
                X, y, X_calib, y_calib, self.calib_size, self.time_ordered, self.random_state
            )
            self.estimator_ = fit_clone(self.estimator, X_tr, y_tr)
            if self.score == "normalized":
                cv = KFold(
                    5, shuffle=not self.time_ordered, random_state=None if self.time_ordered else 0
                )
                oof = cross_val_predict(self.estimator, X_tr, y_tr, cv=cv)
                base = self.scale_estimator if self.scale_estimator is not None else self.estimator
                self.scale_estimator_ = fit_clone(base, X_tr, np.abs(y_tr - oof))
        mu = predict_1d(self.estimator_, X_cal)
        if self.score == "normalized":
            sig = self._sigma(X_cal, fit_floor=True)
            self.calib_scores_ = np.abs(y_cal - mu) / sig
        else:
            self.calib_scores_ = np.abs(y_cal - mu)
        self.n_calib_ = self.calib_scores_.size
        return self

    def _sigma(self, X: ArrayLike, fit_floor: bool = False) -> np.ndarray:
        sig = predict_1d(self.scale_estimator_, X)
        if fit_floor:
            pos = sig[sig > 0]
            self.sigma_floor_ = 1e-3 * (np.median(pos) if pos.size else 1.0)
        return np.maximum(sig, self.sigma_floor_)

    # -------------------------------------------------------------- predict
    def predict(self, X: ArrayLike) -> np.ndarray:
        check_fitted(self, "calib_scores_")
        return predict_1d(self.estimator_, X)

    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray:
        check_fitted(self, "calib_scores_")
        q = conformal_quantile(self.calib_scores_, check_alpha(alpha))
        mu = self.predict(X)
        half = q * self._sigma(X) if self.score == "normalized" else np.full_like(mu, q)
        return as_interval(mu - half, mu + half)


class ConformalizedQuantileRegressor(BaseIntervalRegressor):
    r"""Conformalized quantile regression, CQR (Romano, Patterson & Candès 2019).

    Fits quantile models :math:`\hat q_{\alpha/2}` and
    :math:`\hat q_{1-\alpha/2}` on the proper training set, computes
    :math:`E_i = \max(\hat q_{lo}(X_i) - Y_i,\; Y_i - \hat q_{hi}(X_i))` on the
    calibration set and returns
    :math:`[\hat q_{lo}(x) - \hat Q,\; \hat q_{hi}(x) + \hat Q]` with
    :math:`\hat Q` the conformal quantile of :math:`E` at level :math:`1-\alpha`.

    ``symmetric=False`` calibrates the two tails separately at level
    :math:`1-\alpha/2` each (Romano et al. 2019, Thm. 2), which also controls
    each one-sided miscoverage at :math:`\alpha/2`.

    Parameters
    ----------
    estimator
        A regressor that :func:`reguq.quantile.with_quantile` can configure
        (LightGBM, XGBoost, CatBoost, GPBoost, sklearn GBR/HGB/QuantileRegressor).
        Ignored when ``lower_estimator`` and ``upper_estimator`` are given.
    alpha
        Miscoverage the quantile models are trained for.
    lower_estimator, upper_estimator
        Explicit, already configured quantile regressors (unfitted).
    """

    def __init__(
        self,
        estimator: Any = None,
        *,
        alpha: float = 0.1,
        lower_estimator: Any = None,
        upper_estimator: Any = None,
        symmetric: bool = True,
        calib_size: float | int = 0.25,
        time_ordered: bool = False,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.lower_estimator = lower_estimator
        self.upper_estimator = upper_estimator
        self.symmetric = symmetric
        self.calib_size = calib_size
        self.time_ordered = time_ordered
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_calib: ArrayLike | None = None,
        y_calib: ArrayLike | None = None,
    ) -> ConformalizedQuantileRegressor:
        a = check_alpha(self.alpha)
        explicit = self.lower_estimator is not None and self.upper_estimator is not None
        if not explicit and self.estimator is None:
            raise ValueError("pass `estimator`, or both `lower_estimator` and `upper_estimator`")
        lo = self.lower_estimator if explicit else with_quantile(self.estimator, a / 2)
        hi = self.upper_estimator if explicit else with_quantile(self.estimator, 1 - a / 2)
        X_tr, y_tr, X_cal, y_cal = _resolve_split(
            X, y, X_calib, y_calib, self.calib_size, self.time_ordered, self.random_state
        )
        self.lower_ = fit_clone(lo, X_tr, y_tr)
        self.upper_ = fit_clone(hi, X_tr, y_tr)
        self.median_ = (
            None if explicit else fit_clone(with_quantile(self.estimator, 0.5), X_tr, y_tr)
        )
        q_lo, q_hi = predict_1d(self.lower_, X_cal), predict_1d(self.upper_, X_cal)
        self.calib_lower_scores_ = q_lo - y_cal
        self.calib_upper_scores_ = y_cal - q_hi
        self.calib_scores_ = np.maximum(self.calib_lower_scores_, self.calib_upper_scores_)
        self.n_calib_ = self.calib_scores_.size
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Median prediction if a median model was fitted, else the raw-interval midpoint."""
        check_fitted(self, "calib_scores_")
        if self.median_ is not None:
            return predict_1d(self.median_, X)
        return 0.5 * (predict_1d(self.lower_, X) + predict_1d(self.upper_, X))

    def predict_quantiles(self, X: ArrayLike) -> np.ndarray:
        """Uncalibrated ``(lower, upper)`` quantile predictions, shape ``(n, 2)``."""
        check_fitted(self, "calib_scores_")
        return as_interval(predict_1d(self.lower_, X), predict_1d(self.upper_, X))

    def predict_interval(self, X: ArrayLike, alpha: float | None = None) -> np.ndarray:
        """Calibrated interval. ``alpha`` defaults to the training ``alpha``.

        Calibrating at a different ``alpha`` remains valid (the scores are
        exchangeable for any level) but is usually less efficient.
        """
        check_fitted(self, "calib_scores_")
        a = check_alpha(self.alpha if alpha is None else alpha)
        raw = self.predict_quantiles(X)
        if self.symmetric:
            q = conformal_quantile(self.calib_scores_, a)
            lo, hi = raw[:, 0] - q, raw[:, 1] + q
        else:
            lo = raw[:, 0] - conformal_quantile(self.calib_lower_scores_, a / 2)
            hi = raw[:, 1] + conformal_quantile(self.calib_upper_scores_, a / 2)
        # Crossed quantiles give an empty set; ordering the bounds only enlarges it.
        return as_interval(np.minimum(lo, hi), np.maximum(lo, hi))
