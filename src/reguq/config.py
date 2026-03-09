"""Config coercion and loading helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from .types import OutputConfig, ParamsSourceConfig, SplitConfig


def load_config(config_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)

    path = Path(config_or_path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at top-level.")
    return data


def coerce_split_config(value: Mapping[str, Any] | SplitConfig | None) -> SplitConfig:
    if value is None:
        return SplitConfig()
    if isinstance(value, SplitConfig):
        return value
    cfg = dict(value)
    return SplitConfig(
        test_size=float(cfg.get("test_size", 0.2)),
        shuffle=bool(cfg.get("shuffle", False)),
        random_state=int(cfg.get("random_state", 42)),
    )


def coerce_output_config(value: Mapping[str, Any] | OutputConfig | None) -> OutputConfig:
    if value is None:
        return OutputConfig()
    if isinstance(value, OutputConfig):
        return value
    cfg = dict(value)
    style_overrides = cfg.get("style_overrides", {})
    if style_overrides is None:
        style_overrides = {}
    if not isinstance(style_overrides, Mapping):
        raise ValueError("output.style_overrides must be a mapping when provided.")

    return OutputConfig(
        output_dir=cfg.get("output_dir"),
        export_excel=bool(cfg.get("export_excel", False)),
        export_plots=bool(cfg.get("export_plots", False)),
        embed_excel_charts=bool(cfg.get("embed_excel_charts", False)),
        show_inline_plots=bool(cfg.get("show_inline_plots", False)),
        chart_detail_level=str(cfg.get("chart_detail_level", "detailed")),
        legend_position=str(cfg.get("legend_position", "upper right")),
        style_overrides=dict(style_overrides),
        save_json=bool(cfg.get("save_json", True)),
        run_id=cfg.get("run_id"),
    )


def coerce_params_source(value: Mapping[str, Any] | ParamsSourceConfig | None) -> ParamsSourceConfig:
    if value is None:
        return ParamsSourceConfig()
    if isinstance(value, ParamsSourceConfig):
        return value
    cfg = dict(value)
    return ParamsSourceConfig(
        mode=str(cfg.get("mode", "load_or_tune")),
        params=dict(cfg.get("params", {})),
        params_path=cfg.get("params_path"),
        tuning_config=dict(cfg.get("tuning_config", {})),
    )


def as_plain_dict(config: OutputConfig | ParamsSourceConfig | SplitConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    return asdict(config)
