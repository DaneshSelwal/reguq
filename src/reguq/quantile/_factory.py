"""Turn a point regressor into a quantile regressor at level ``q``.

The 0.2 code set any parameter called ``alpha`` - which for a squared-error
GradientBoostingRegressor does nothing, and for Ridge/Lasso changes the
regularisation strength. Here every supported backend is mapped explicitly
to its own quantile objective, and anything unknown raises.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import clone


def _gbr(est: Any, q: float) -> dict[str, Any]:
    return {"loss": "quantile", "alpha": q}


def _hgb(est: Any, q: float) -> dict[str, Any]:
    return {"loss": "quantile", "quantile": q}


def _sk_linear(est: Any, q: float) -> dict[str, Any]:
    return {"quantile": q}


def _lgbm(est: Any, q: float) -> dict[str, Any]:
    return {"objective": "quantile", "alpha": q}


def _xgb(est: Any, q: float) -> dict[str, Any]:
    return {"objective": "reg:quantileerror", "quantile_alpha": q}


def _catboost(est: Any, q: float) -> dict[str, Any]:
    return {"loss_function": f"Quantile:alpha={q}"}


def _gpboost(est: Any, q: float) -> dict[str, Any]:
    return {"objective": "quantile", "alpha": q}


# Keyed by class name so no backend has to be imported to check.
_QUANTILE_PARAMS: dict[str, Callable[[Any, float], dict[str, Any]]] = {
    "GradientBoostingRegressor": _gbr,
    "HistGradientBoostingRegressor": _hgb,
    "QuantileRegressor": _sk_linear,
    "LGBMRegressor": _lgbm,
    "XGBRegressor": _xgb,
    "CatBoostRegressor": _catboost,
    "GPBoostRegressor": _gpboost,
}


def supports_quantile(estimator: Any) -> bool:
    """True if :func:`with_quantile` knows how to configure ``estimator``."""
    return type(estimator).__name__ in _QUANTILE_PARAMS or hasattr(estimator, "set_quantile")


def with_quantile(estimator: Any, q: float) -> Any:
    """Return an unfitted clone of ``estimator`` configured to predict quantile ``q``.

    Estimators may also opt in by defining ``set_quantile(q)`` returning a
    configured copy (used by reguq's own TabNet / PGBM wrappers).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile level must lie in (0, 1); got {q}")
    if hasattr(estimator, "set_quantile"):
        return estimator.set_quantile(q)
    name = type(estimator).__name__
    if name not in _QUANTILE_PARAMS:
        raise TypeError(
            f"don't know how to make {name} predict a quantile; pass explicit lower/upper "
            f"quantile estimators, or one of: {sorted(_QUANTILE_PARAMS)}"
        )
    est = clone(estimator)
    est.set_params(**_QUANTILE_PARAMS[name](est, q))
    return est
