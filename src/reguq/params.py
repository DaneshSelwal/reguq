"""Hyperparameter source resolution (load-or-tune/defaults)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import coerce_params_source
from .tuning import tune_single_model
from .types import ParamsSourceConfig


def _load_params_file(path: str | Path) -> dict[str, dict[str, Any]]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError("Parameter file must contain a model->params mapping.")

    return {str(k): dict(v) for k, v in loaded.items()}


def resolve_model_params(
    models: list[str],
    params_source: ParamsSourceConfig | Mapping[str, Any] | None,
    X_train,
    y_train,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Resolve params for requested models.

    Returns (resolved_params, tuned_params_only).
    """

    source = coerce_params_source(params_source)
    mode = source.mode
    if mode not in {"load_or_tune", "load_only", "defaults"}:
        raise ValueError("params_source.mode must be one of: load_or_tune, load_only, defaults")

    resolved: dict[str, dict[str, Any]] = {}
    tuned: dict[str, dict[str, Any]] = {}

    loaded_params = dict(source.params)
    if source.params_path:
        loaded_params.update(_load_params_file(source.params_path))

    for model_id in models:
        if mode == "defaults":
            resolved[model_id] = {}
            continue

        if model_id in loaded_params:
            resolved[model_id] = dict(loaded_params[model_id])
            continue

        if mode == "load_only":
            raise ValueError(
                f"Missing parameters for '{model_id}' in load_only mode. "
                "Provide params or switch to load_or_tune."
            )

        tuned_params, _ = tune_single_model(
            model_id=model_id,
            X_train=X_train,
            y_train=y_train,
            tuning_config=source.tuning_config,
        )
        resolved[model_id] = tuned_params
        tuned[model_id] = tuned_params

    return resolved, tuned
