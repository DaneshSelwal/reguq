"""reguq package - Regression Uncertainty Quantification."""

from __future__ import annotations

from .api import (
    run_conformal_standard,
    run_conformal_advanced,
    run_from_config,
    run_probabilistic,
    run_probabilistic_advanced,
    run_quantile,
    run_tuning,
    run_explainability,
    explain_shap,
    explain_lime,
    explain_interpret,
    CARDRegressor,
    IBUGRegressor,
)
from .colab import bootstrap_colab_environment
from .types import (
    ConformalResult,
    DataBundle,
    OutputConfig,
    ParamsSourceConfig,
    PhaseResult,
    PipelineRunResult,
    SplitConfig,
    TuningResult,
)

__all__ = [
    # Core pipeline functions
    "run_tuning",
    "run_quantile",
    "run_probabilistic",
    "run_conformal_standard",
    "run_from_config",
    "bootstrap_colab_environment",
    # Advanced methods
    "run_conformal_advanced",
    "run_probabilistic_advanced",
    "run_explainability",
    # CARD & IBUG classes
    "CARDRegressor",
    "IBUGRegressor",
    # Explainability functions
    "explain_shap",
    "explain_lime",
    "explain_interpret",
    # Types
    "SplitConfig",
    "OutputConfig",
    "ParamsSourceConfig",
    "DataBundle",
    "PhaseResult",
    "TuningResult",
    "ConformalResult",
    "PipelineRunResult",
]
