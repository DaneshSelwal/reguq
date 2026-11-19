"""Lazy model registry: build point regressors by name without importing every backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering ``factory(random_state=None, **params) -> estimator``."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name.lower()] = fn
        return fn

    return deco


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def make_model(
    name: str, params: dict[str, Any] | None = None, random_state: int | None = None
) -> Any:
    """Instantiate the registered model ``name`` with ``params``.

    ``random_state`` is passed to every backend under its own parameter
    name, so seeds are never hard-coded.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}")
    return _REGISTRY[key](random_state=random_state, **(params or {}))


def _require(module: str, extra: str) -> Any:
    from ..probabilistic._base import require

    return require(module, extra)


@register_model("linear")
def _linear(random_state: int | None = None, **p: Any) -> Any:
    from sklearn.linear_model import Ridge

    return Ridge(**{"alpha": 1e-6, **p})


@register_model("randomforest")
def _rf(random_state: int | None = None, **p: Any) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(random_state=random_state, **{"n_estimators": 300, **p})


@register_model("extratrees")
def _et(random_state: int | None = None, **p: Any) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(random_state=random_state, **{"n_estimators": 300, **p})


@register_model("gradientboosting")
def _gbr(random_state: int | None = None, **p: Any) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(random_state=random_state, **p)


@register_model("histgradientboosting")
def _hgb(random_state: int | None = None, **p: Any) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(random_state=random_state, **p)


@register_model("lightgbm")
def _lgbm(random_state: int | None = None, **p: Any) -> Any:
    lgb = _require("lightgbm", "boosting")
    return lgb.LGBMRegressor(random_state=random_state, **{"verbose": -1, **p})


@register_model("xgboost")
def _xgb(random_state: int | None = None, **p: Any) -> Any:
    xgb = _require("xgboost", "boosting")
    return xgb.XGBRegressor(random_state=random_state, **p)


@register_model("catboost")
def _cat(random_state: int | None = None, **p: Any) -> Any:
    cb = _require("catboost", "boosting")
    return cb.CatBoostRegressor(random_seed=random_state, **{"verbose": 0, **p})


@register_model("gpboost")
def _gpb(random_state: int | None = None, **p: Any) -> Any:
    gpb = _require("gpboost", "gpboost")
    return gpb.GPBoostRegressor(random_state=random_state, **{"verbose": -1, **p})


@register_model("tabnet")
def _tabnet(random_state: int | None = None, **p: Any) -> Any:
    from ..quantile.tabnet import TabNetRegressor

    return TabNetRegressor(random_state=0 if random_state is None else random_state, **p)


__all__ = ["available_models", "make_model", "register_model"]
