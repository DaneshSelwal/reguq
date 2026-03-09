"""Shared constants for reguq."""

from __future__ import annotations

PHASE_TUNING = "tuning"
PHASE_QUANTILE = "quantile"
PHASE_PROBABILISTIC = "probabilistic"
PHASE_CONFORMAL_STANDARD = "conformal_standard"

ALL_PHASES = (
    PHASE_TUNING,
    PHASE_QUANTILE,
    PHASE_PROBABILISTIC,
    PHASE_CONFORMAL_STANDARD,
)

CORE_MODELS = (
    "lightgbm",
    "xgboost",
    "catboost",
    "ngboost",
    "pgbm",
)

DEFAULT_QUANTILES = (0.05, 0.95)
DEFAULT_ALPHA = 0.1
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
