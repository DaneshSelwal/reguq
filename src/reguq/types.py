"""Public result and config types for reguq."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SplitConfig:
    test_size: float = 0.2
    shuffle: bool = False
    random_state: int = 42


@dataclass
class OutputConfig:
    output_dir: str | Path | None = None
    export_excel: bool = False
    export_plots: bool = False
    save_json: bool = True
    run_id: str | None = None


@dataclass
class ParamsSourceConfig:
    mode: str = "load_or_tune"
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    params_path: str | Path | None = None
    tuning_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataBundle:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_columns: list[str]


@dataclass
class PhaseResult:
    phase: str
    predictions: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class TuningResult:
    phase: str
    best_params: dict[str, dict[str, Any]]
    summary: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class ConformalResult:
    phase: str
    methods: dict[str, PhaseResult]
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class PipelineRunResult:
    run_id: str
    output_dir: Path | None
    results: dict[str, Any]
