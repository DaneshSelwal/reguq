"""Pipeline configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration of one pipeline run.

    Attributes
    ----------
    train_path, test_path
        CSV files. Column names are whitespace-stripped on load.
    target
        Target column (default: the last column of the training file).
    test_target
        Only when the test file names the target differently; this means the
        test measures a different quantity and is therefore explicit.
    time_ordered
        Never shuffle; calibration blocks are taken from the end of training.
    alpha
        Miscoverage level of every interval (``1 - alpha`` coverage).
    models, methods, model_free_methods
        Registered model names (:func:`reguq.models.available_models`) and
        method names (:data:`reguq.pipeline.METHODS` / ``MODEL_FREE``).
    tuning
        Keyword arguments for :func:`reguq.tuning.tune`, or ``None`` to skip.
    params
        Fixed hyper-parameters per model (skips tuning for that model).
    method_params
        Extra constructor arguments per method.
    output_dir, export_excel, export_plots
        Where and what to export (see :func:`reguq.pipeline.export`).
    """

    train_path: str
    test_path: str
    target: str | None = None  # default: last column
    test_target: str | None = None  # only if the test file names the target differently
    features: list[str] | None = None
    time_ordered: bool = False
    alpha: float = 0.1
    calib_size: float = 0.25
    random_state: int = 0
    models: list[str] = field(
        default_factory=lambda: [
            "lightgbm",
            "xgboost",
            "catboost",
            "gradientboosting",
            "randomforest",
        ]
    )
    methods: list[str] = field(
        default_factory=lambda: [
            "split",
            "normalized",
            "cqr",
            "cv_plus",
            "jab",
            "multi_quantile",
            "ibug",
        ]
    )
    model_free_methods: list[str] = field(default_factory=list)  # e.g. ["ngboost", "qrf", "hcm"]
    tuning: dict[str, Any] | None = field(default_factory=lambda: {"n_trials": 30, "n_splits": 5})
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    method_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_dir: str | None = None
    export_excel: bool = False
    export_plots: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail fast on invalid settings, before any data is loaded or model is fitted."""
        from ..models import available_models
        from .methods import METHODS, MODEL_FREE

        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {self.alpha}")
        if not 0.0 < float(self.calib_size) < 1.0:
            raise ValueError(f"calib_size must lie in (0, 1); got {self.calib_size}")
        unknown = [m for m in self.models if m.lower() not in available_models()]
        if unknown:
            raise KeyError(f"unknown models {unknown}; available: {available_models()}")
        bad = [m for m in self.methods if m not in METHODS]
        if bad:
            raise KeyError(f"unknown methods {bad}; available: {sorted(METHODS)}")
        bad = [m for m in self.model_free_methods if m not in MODEL_FREE]
        if bad:
            raise KeyError(f"unknown model-free methods {bad}; available: {sorted(MODEL_FREE)}")

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            from ..probabilistic._base import require

            data = require("yaml", "report").safe_load(text)
        else:
            data = json.loads(text)
        return cls(**data)
