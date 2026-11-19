"""Quantile regression: backend objectives, multi-quantile models, QRF, TabNet."""

from __future__ import annotations

from ._factory import supports_quantile, with_quantile
from .multi import (
    DEFAULT_QUANTILES,
    MultiQuantileRegressor,
    QuantileForestRegressor,
    crossing_rate,
    rearrange_quantiles,
)
from .tabnet import TabNetRegressor

__all__ = [
    "DEFAULT_QUANTILES",
    "MultiQuantileRegressor",
    "QuantileForestRegressor",
    "TabNetRegressor",
    "crossing_rate",
    "rearrange_quantiles",
    "supports_quantile",
    "with_quantile",
]
