"""Public API entry points."""

from __future__ import annotations

from .colab import bootstrap_colab_environment
from .conformal_standard import run_conformal_standard
from .conformal_advanced import run_conformal_advanced
from .probabilistic import run_probabilistic
from .probabilistic_advanced import run_probabilistic_advanced, CARDRegressor, IBUGRegressor
from .quantile import run_quantile
from .runner import run_from_config
from .tuning import run_tuning
from .explainability import run_explainability, explain_shap, explain_lime, explain_interpret

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
]
