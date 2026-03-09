"""Standard conformal prediction phase (MAPIE/PUNCC with fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .charts import generate_conformal_charts
from .config import coerce_output_config
from .constants import DEFAULT_ALPHA, PHASE_CONFORMAL_STANDARD
from .data import prepare_data_bundle
from .export import embed_images_in_excel, write_conformal_excel, write_json
from .metrics import interval_metrics, regression_metrics
from .params import resolve_model_params
from .types import ConformalResult, OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(intervals)
    if arr.ndim == 3:
        # Common MAPIE shape: (n_samples, 2, n_alpha)
        lower = arr[:, 0, 0]
        upper = arr[:, 1, 0]
        return np.asarray(lower).ravel(), np.asarray(upper).ravel()

    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.asarray(arr[:, 0]).ravel(), np.asarray(arr[:, 1]).ravel()

    raise ValueError(f"Unsupported interval shape: {arr.shape}")


def _manual_split_conformal(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
    calibration_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=calibration_size,
        shuffle=False,
        random_state=random_state,
    )
    estimator.fit(X_fit, y_fit)
    y_cal_pred = np.asarray(estimator.predict(X_cal)).ravel()
    scores = np.abs(y_cal.to_numpy() - y_cal_pred)
    q = np.quantile(scores, 1 - alpha, method="higher")

    y_pred = np.asarray(estimator.predict(X_test)).ravel()
    y_lower = y_pred - q
    y_upper = y_pred + q
    return y_pred, y_lower, y_upper


def _predict_mapie(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
    method: str,
):
    from mapie.regression import MapieRegressor

    mapie_model = MapieRegressor(estimator=estimator, method=method, cv="split")
    mapie_model.fit(X_train, y_train)
    y_pred, intervals = mapie_model.predict(X_test, alpha=alpha)
    y_lower, y_upper = _extract_interval_bounds(intervals)
    return np.asarray(y_pred).ravel(), y_lower, y_upper


def _predict_puncc(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
):
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import SplitCP

    predictor = BasePredictor(estimator)
    cp = SplitCP(predictor)
    cp.fit(X_train.to_numpy(), y_train.to_numpy())
    outputs = cp.predict(X_test.to_numpy(), alpha=alpha)

    if isinstance(outputs, tuple) and len(outputs) >= 2:
        y_pred = np.asarray(outputs[0]).ravel()
        intervals = outputs[1]
    else:
        raise ValueError("Unexpected PUNCC predict output format.")

    y_lower, y_upper = _extract_interval_bounds(intervals)
    return y_pred, y_lower, y_upper


def _run_method(
    method_name: str,
    model_ids: list[str],
    model_params: dict[str, dict[str, Any]],
    bundle,
    alpha: float,
    mapie_method: str,
    calibration_size: float,
    random_state: int,
) -> PhaseResult:
    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))
        estimator = model_registry.build_estimator(
            model_id=model_id,
            phase=PHASE_CONFORMAL_STANDARD,
            params=params,
        )

        backend = method_name
        try:
            if method_name == "mapie":
                y_pred, y_lower, y_upper = _predict_mapie(
                    estimator=estimator,
                    X_train=bundle.X_train,
                    y_train=bundle.y_train,
                    X_test=bundle.X_test,
                    alpha=alpha,
                    method=mapie_method,
                )
            elif method_name == "puncc":
                y_pred, y_lower, y_upper = _predict_puncc(
                    estimator=estimator,
                    X_train=bundle.X_train,
                    y_train=bundle.y_train,
                    X_test=bundle.X_test,
                    alpha=alpha,
                )
            else:
                raise ValueError(f"Unknown conformal method '{method_name}'")
        except Exception:
            backend = "manual_fallback"
            y_pred, y_lower, y_upper = _manual_split_conformal(
                estimator=estimator,
                X_train=bundle.X_train,
                y_train=bundle.y_train,
                X_test=bundle.X_test,
                alpha=alpha,
                calibration_size=calibration_size,
                random_state=random_state,
            )

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

        row = {"model": model_id, "method": method_name, "backend": backend, "alpha": alpha}
        row.update(regression_metrics(y_true=y_true, y_pred=y_pred))
        row.update(interval_metrics(y_true=y_true, y_lower=y_lower, y_upper=y_upper))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    return PhaseResult(
        phase=f"{PHASE_CONFORMAL_STANDARD}_{method_name}",
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )


def run_conformal_standard(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    conformal_config: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
) -> ConformalResult:
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_CONFORMAL_STANDARD)
    output_cfg = coerce_output_config(output_config)

    cfg = dict(conformal_config or {})
    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    mapie_method = str(cfg.get("mapie_method", "plus"))
    methods = list(cfg.get("methods", ["mapie", "puncc"]))
    calibration_size = float(cfg.get("calibration_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    if not (0 < alpha < 1):
        raise ValueError("conformal alpha must satisfy 0 < alpha < 1")

    model_params, tuned_params = resolve_model_params(
        models=model_ids,
        params_source=params_source,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
    )

    method_results: dict[str, PhaseResult] = {}
    for method_name in methods:
        method_results[method_name] = _run_method(
            method_name=method_name,
            model_ids=model_ids,
            model_params=model_params,
            bundle=bundle,
            alpha=alpha,
            mapie_method=mapie_method,
            calibration_size=calibration_size,
            random_state=random_state,
        )

    result = ConformalResult(
        phase=PHASE_CONFORMAL_STANDARD,
        methods=method_results,
        artifacts=[],
    )

    if output_cfg.output_dir is not None:
        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chart_result = None
        if output_cfg.export_plots or output_cfg.embed_excel_charts or output_cfg.show_inline_plots:
            chart_result = generate_conformal_charts(
                conformal_result=result,
                output_cfg=output_cfg,
                output_dir=output_dir,
            )
        if output_cfg.export_plots and chart_result is not None:
            result.artifacts.extend(chart_result.image_paths)

        excel_path = output_dir / "conformal_standard.xlsx"
        if output_cfg.export_excel:
            result.artifacts.append(write_conformal_excel(result, excel_path))
            if output_cfg.embed_excel_charts and chart_result is not None and chart_result.images_by_sheet:
                embed_images_in_excel(workbook_path=excel_path, images_by_sheet=chart_result.images_by_sheet)

        if output_cfg.save_json and tuned_params:
            result.artifacts.append(write_json(tuned_params, output_dir / "conformal_tuned_params.json"))

    return result
