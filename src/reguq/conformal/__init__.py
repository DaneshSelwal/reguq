"""Conformal prediction for regression.

Exchangeable data
    :class:`SplitConformalRegressor`, :class:`ConformalizedQuantileRegressor`,
    :class:`CrossConformalRegressor` (CV+ / Jackknife+),
    :class:`JackknifePlusAfterBootstrapRegressor`.
Distribution drift
    :class:`NexCPRegressor`, :class:`OnlineConformalRegressor`
    (ACI, DtACI/FACI, SF-OGD, SAOCP, online NexCP).
Small alpha
    :class:`ExtremeValueConformalRegressor` (GPD tail; no finite-sample guarantee).
Data-dependent coverage
    :class:`AdaptiveCoveragePolicyRegressor` (e-value based, post-hoc guarantee).
"""

from __future__ import annotations

from ._quantile import (
    conformal_lower_quantile,
    conformal_quantile,
    conformal_rank,
    conformal_upper_quantile,
    weighted_conformal_quantile,
    weighted_lower_quantile,
)
from .acp import AdaptiveCoveragePolicyRegressor
from .extreme import ExtremeValueConformalRegressor, gpd_tail_quantile
from .nonexchangeable import NexCPRegressor, exponential_weights
from .online import (
    ACI,
    FACI,
    SAOCP,
    DtACI,
    NexCPOnline,
    OnlineConformalRegressor,
    OnlineTracker,
    ScaleFreeOGD,
)
from .resampling import CrossConformalRegressor, JackknifePlusAfterBootstrapRegressor
from .split import ConformalizedQuantileRegressor, SplitConformalRegressor

__all__ = [
    "ACI",
    "FACI",
    "SAOCP",
    "AdaptiveCoveragePolicyRegressor",
    "ConformalizedQuantileRegressor",
    "CrossConformalRegressor",
    "DtACI",
    "ExtremeValueConformalRegressor",
    "JackknifePlusAfterBootstrapRegressor",
    "NexCPOnline",
    "NexCPRegressor",
    "OnlineConformalRegressor",
    "OnlineTracker",
    "ScaleFreeOGD",
    "SplitConformalRegressor",
    "conformal_lower_quantile",
    "conformal_quantile",
    "conformal_rank",
    "conformal_upper_quantile",
    "exponential_weights",
    "gpd_tail_quantile",
    "weighted_conformal_quantile",
    "weighted_lower_quantile",
]
