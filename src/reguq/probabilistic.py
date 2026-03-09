"""Probabilistic regression phase."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import coerce_output_config
from .constants import DEFAULT_ALPHA, PHASE_PROBABILISTIC
from .data import prepare_data_bundle
from .export import save_interval_plot, write_json, write_phase_excel
from .metrics import gaussian_crps, gaussian_nll, interval_metrics, regression_metrics
from .params import resolve_model_params
from .types import OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def _safe_sigma(values: np.ndarray, fallback: float = 1.0) -> np.ndarray:
    sigma = np.asarray(values, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, fallback)
    sigma = np.maximum(sigma, 1e-8)
    return sigma


def _mean_std_from_samples(samples: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray(samples)
    if arr.ndim != 2:
        return None

    if arr.shape[0] == n_rows:
        mean = np.mean(arr, axis=1)
        std = np.std(arr, axis=1, ddof=1)
        return mean, std

    if arr.shape[1] == n_rows:
        arr = arr.T
        mean = np.mean(arr, axis=1)
        std = np.std(arr, axis=1, ddof=1)
        return mean, std

    return None


def _predict_distribution(estimator, model_id: str, X_train, y_train, X_test) -> tuple[np.ndarray, np.ndarray]:
    if model_id == "ngboost" and hasattr(estimator, "pred_dist"):
        dist = estimator.pred_dist(X_test)
        mean = np.asarray(dist.loc).ravel()
        std = np.asarray(dist.scale).ravel()
        return mean, _safe_sigma(std)

    if hasattr(estimator, "predict_dist"):
        try:
            samples = estimator.predict_dist(X_test, n_forecasts=200)
            converted = _mean_std_from_samples(np.asarray(samples), n_rows=len(X_test))
            if converted is not None:
                mean, std = converted
                return np.asarray(mean).ravel(), _safe_sigma(np.asarray(std).ravel())
        except Exception:
            pass

    mean = np.asarray(estimator.predict(X_test)).ravel()
    residuals = np.asarray(y_train).ravel() - np.asarray(estimator.predict(X_train)).ravel()
    sigma = np.full_like(mean, fill_value=max(float(np.std(residuals, ddof=1)), 1e-8), dtype=float)
    return mean, sigma


def run_probabilistic(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> PhaseResult:
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_PROBABILISTIC)
    output_cfg = coerce_output_config(output_config)

    if not (0 < alpha < 1):
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    model_params, tuned_params = resolve_model_params(
        models=model_ids,
        params_source=params_source,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
    )

    z_low = norm.ppf(alpha / 2.0)
    z_high = norm.ppf(1.0 - alpha / 2.0)

    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))
        estimator = model_registry.build_estimator(
            model_id=model_id,
            phase=PHASE_PROBABILISTIC,
            params=params,
        )
        estimator.fit(bundle.X_train, bundle.y_train)

        mean, sigma = _predict_distribution(
            estimator=estimator,
            model_id=model_id,
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            X_test=bundle.X_test,
        )
        y_true = bundle.y_test.to_numpy()

        y_lower = mean + z_low * sigma
        y_upper = mean + z_high * sigma

        pred_df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": mean,
                "y_std": sigma,
                "y_lower": y_lower,
                "y_upper": y_upper,
            }
        )
        predictions[model_id] = pred_df

        row = {"model": model_id, "alpha": alpha}
        row.update(regression_metrics(y_true=y_true, y_pred=mean))
        row.update(interval_metrics(y_true=y_true, y_lower=y_lower, y_upper=y_upper))
        row.update(
            {
                "nll": gaussian_nll(y_true=y_true, mean=mean, std=sigma),
                "crps": gaussian_crps(y_true=y_true, mean=mean, std=sigma),
            }
        )
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    result = PhaseResult(
        phase=PHASE_PROBABILISTIC,
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )

    if output_cfg.output_dir is not None:
        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_cfg.export_excel:
            result.artifacts.append(write_phase_excel(result, output_dir / "probabilistic.xlsx"))

        if output_cfg.save_json and tuned_params:
            result.artifacts.append(write_json(tuned_params, output_dir / "probabilistic_tuned_params.json"))

        if output_cfg.export_plots:
            for model_id, pred_df in predictions.items():
                result.artifacts.append(
                    save_interval_plot(
                        pred_df,
                        output_dir / f"probabilistic_{model_id}.png",
                        title=f"Probabilistic Intervals - {model_id}",
                    )
                )

    return result
