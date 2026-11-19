"""End-to-end pipeline: load -> tune on train only -> fit UQ methods -> evaluate on test.

Design rules (each fixes a v1.2 problem):

* the test set is used **once**, for final evaluation - never for tuning,
  early stopping or calibration;
* hyper-parameters are tuned once per model and shared by every method;
* a method that fails is reported with its error message - nothing silently
  falls back to a different method under the same label;
* every method is evaluated with the same metric functions.
"""

from __future__ import annotations

from .cli import main
from .config import PipelineConfig
from .data import load_data
from .evaluate import evaluate
from .export import export
from .methods import METHODS, MODEL_FREE
from .runner import PipelineResult, run_pipeline

__all__ = [
    "METHODS",
    "MODEL_FREE",
    "PipelineConfig",
    "PipelineResult",
    "evaluate",
    "export",
    "load_data",
    "main",
    "run_pipeline",
]
