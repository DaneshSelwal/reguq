"""Probabilistic (distributional) regression.

Every model returns a :class:`GaussianPrediction` or :class:`SamplePrediction`
from ``predict_dist``. Backends are optional extras and imported lazily.
"""

from __future__ import annotations

from ._base import BaseDistributionalRegressor, StdEstimator
from ._dist import GaussianPrediction, SamplePrediction
from .boosting import IBUGRegressor, NGBoostRegressor, PGBMRegressor
from .neural import CARDRegressor, HCMRegressor
from .treeffuser import TreeffuserRegressor

__all__ = [
    "BaseDistributionalRegressor",
    "CARDRegressor",
    "GaussianPrediction",
    "HCMRegressor",
    "IBUGRegressor",
    "NGBoostRegressor",
    "PGBMRegressor",
    "SamplePrediction",
    "StdEstimator",
    "TreeffuserRegressor",
]
