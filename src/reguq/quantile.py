"""Quantile regression phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import coerce_output_config
from .constants import DEFAULT_QUANTILES, PHASE_QUANTILE
from .data import prepare_data_bundle
from .export import save_interval_plot, write_json, write_phase_excel
from .metrics import interval_metrics, regression_metrics
from .params import resolve_model_params
from .types import OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def run_quantile(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
    quantiles: tuple[float, float] = DEFAULT_QUANTILES,
) -> PhaseResult:
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_QUANTILE)
    output_cfg = coerce_output_config(output_config)

    q_low, q_high = quantiles
    if not (0 < q_low < q_high < 1):
        raise ValueError("quantiles must satisfy 0 < q_low < q_high < 1")

    model_params, tuned_params = resolve_model_params(
        models=model_ids,
        params_source=params_source,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
    )

    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))

        lower_model = model_registry.build_estimator(
            model_id=model_id,
            phase=PHASE_QUANTILE,
            params=params,
            quantile=q_low,
        )
        upper_model = model_registry.build_estimator(
            model_id=model_id,
            phase=PHASE_QUANTILE,
            params=params,
            quantile=q_high,
        )

        lower_model.fit(bundle.X_train, bundle.y_train)
        upper_model.fit(bundle.X_train, bundle.y_train)

        y_low = np.asarray(lower_model.predict(bundle.X_test)).ravel()
        y_high = np.asarray(upper_model.predict(bundle.X_test)).ravel()
        y_lower = np.minimum(y_low, y_high)
        y_upper = np.maximum(y_low, y_high)
        y_pred = (y_lower + y_upper) / 2.0
        y_true = bundle.y_test.to_numpy()

        pred_df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "y_lower": y_lower,
                "y_upper": y_upper,
            }
        )
        predictions[model_id] = pred_df

        row = {"model": model_id, "quantile_low": q_low, "quantile_high": q_high}
        row.update(regression_metrics(y_true=y_true, y_pred=y_pred))
        row.update(interval_metrics(y_true=y_true, y_lower=y_lower, y_upper=y_upper))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    result = PhaseResult(
        phase=PHASE_QUANTILE,
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )

    if output_cfg.output_dir is not None:
        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_cfg.export_excel:
            result.artifacts.append(write_phase_excel(result, output_dir / "quantile.xlsx"))

        if output_cfg.save_json and tuned_params:
            result.artifacts.append(write_json(tuned_params, output_dir / "quantile_tuned_params.json"))

        if output_cfg.export_plots:
            for model_id, pred_df in predictions.items():
                result.artifacts.append(
                    save_interval_plot(
                        pred_df,
                        output_dir / f"quantile_{model_id}.png",
                        title=f"Quantile Intervals - {model_id}",
                    )
                )

    return result
