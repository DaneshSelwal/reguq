"""Config-driven pipeline runner and CLI."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .config import coerce_output_config, load_config
from .constants import (
    ALL_PHASES,
    PHASE_CONFORMAL_STANDARD,
    PHASE_PROBABILISTIC,
    PHASE_QUANTILE,
    PHASE_TUNING,
)
from .conformal_standard import run_conformal_standard
from .export import make_run_id
from .probabilistic import run_probabilistic
from .quantile import run_quantile
from .tuning import run_tuning
from .types import PipelineRunResult


def _extract_data_input(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("train", "test", "data", "train_df", "test_df", "data_df"):
        if key in data_cfg:
            result[key] = data_cfg[key]
    for key in ("train_path", "test_path", "data_path"):
        if key in data_cfg:
            result[key] = data_cfg[key]
    return result


def run_from_config(config_or_path: Mapping[str, Any] | str | Path) -> PipelineRunResult:
    config = load_config(config_or_path)

    data_cfg = dict(config.get("data", {}))
    target_col = data_cfg.get("target_col") or config.get("target_col")
    if not target_col:
        raise ValueError("target_col is required (config.data.target_col or top-level target_col).")

    data_input = _extract_data_input(data_cfg)
    if not data_input:
        raise ValueError("Config data section must define train/test or data source.")

    split_config = data_cfg.get("split") or config.get("split")
    models = config.get("models")

    phases = list(config.get("phases", list(ALL_PHASES)))
    invalid_phases = [p for p in phases if p not in ALL_PHASES]
    if invalid_phases:
        raise ValueError(f"Unsupported phases: {invalid_phases}")

    output_cfg = coerce_output_config(config.get("output"))
    output_map = config.get("output", {}) if isinstance(config.get("output"), dict) else {}

    run_id = output_cfg.run_id or make_run_id()
    output_cfg.run_id = run_id
    if output_cfg.output_dir is None:
        output_cfg.output_dir = Path("outputs") / run_id
    if "export_excel" not in output_map:
        output_cfg.export_excel = True

    output_dir = Path(output_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params_source = deepcopy(config.get("params_source", {}))
    if not isinstance(params_source, dict):
        raise ValueError("params_source must be a mapping when provided.")
    params_source.setdefault("mode", "load_or_tune")
    params_source.setdefault("params", {})
    if not isinstance(params_source["params"], dict):
        raise ValueError("params_source.params must be a mapping")

    results: dict[str, Any] = {}

    if PHASE_TUNING in phases:
        tuning_result = run_tuning(
            data=data_input,
            target_col=target_col,
            models=models,
            tuning_config=config.get("tuning"),
            output_config=output_cfg,
            split_config=split_config,
        )
        results[PHASE_TUNING] = tuning_result
        params_source["params"].update(tuning_result.best_params)

    if PHASE_QUANTILE in phases:
        quantile_result = run_quantile(
            data=data_input,
            target_col=target_col,
            models=models,
            params_source=params_source,
            output_config=output_cfg,
            split_config=split_config,
            quantiles=tuple(config.get("quantile", {}).get("quantiles", (0.05, 0.95))),
        )
        results[PHASE_QUANTILE] = quantile_result

    if PHASE_PROBABILISTIC in phases:
        probabilistic_cfg = config.get("probabilistic", {})
        probabilistic_result = run_probabilistic(
            data=data_input,
            target_col=target_col,
            models=models,
            params_source=params_source,
            output_config=output_cfg,
            split_config=split_config,
            alpha=float(probabilistic_cfg.get("alpha", 0.1)),
        )
        results[PHASE_PROBABILISTIC] = probabilistic_result

    if PHASE_CONFORMAL_STANDARD in phases:
        conformal_result = run_conformal_standard(
            data=data_input,
            target_col=target_col,
            models=models,
            params_source=params_source,
            conformal_config=config.get("conformal_standard"),
            output_config=output_cfg,
            split_config=split_config,
        )
        results[PHASE_CONFORMAL_STANDARD] = conformal_result

    return PipelineRunResult(run_id=run_id, output_dir=output_dir, results=results)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reguq pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_result = run_from_config(args.config)
    print(f"Run completed. run_id={run_result.run_id}, output_dir={run_result.output_dir}")
    print(f"Completed phases: {', '.join(run_result.results.keys())}")
    return 0
