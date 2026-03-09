"""Hyperparameter tuning workflow."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from .config import coerce_output_config
from .constants import PHASE_TUNING
from .data import prepare_data_bundle
from .export import save_interval_plot, write_json, write_tuning_excel
from .metrics import regression_metrics
from .types import OutputConfig, SplitConfig, TuningResult
import reguq.registry as model_registry


def _load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for tuning. Install with `pip install optuna==4.6.0`."
        ) from exc
    return optuna


def tune_single_model(
    model_id: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    tuning_config = dict(tuning_config or {})
    optuna = _load_optuna()

    n_trials = int(tuning_config.get("n_trials", 20))
    scoring = str(tuning_config.get("scoring", "neg_root_mean_squared_error"))
    cv_splits = int(tuning_config.get("cv", 3))
    timeout = tuning_config.get("timeout")
    random_state = int(tuning_config.get("random_state", 42))

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def objective(trial: Any) -> float:
        params = model_registry.suggest_hyperparameters(trial, model_id)
        estimator = model_registry.build_estimator(model_id, phase=PHASE_TUNING, params=params)
        scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return dict(study.best_params), float(study.best_value)


def run_tuning(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    tuning_config: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
) -> TuningResult:
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models, phase=PHASE_TUNING)
    output_cfg = coerce_output_config(output_config)

    best_params: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        params, best_score = tune_single_model(
            model_id=model_id,
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            tuning_config=tuning_config,
        )
        best_params[model_id] = params

        estimator = model_registry.build_estimator(model_id, phase=PHASE_TUNING, params=params)
        estimator.fit(bundle.X_train, bundle.y_train)
        y_pred = np.asarray(estimator.predict(bundle.X_test)).ravel()
        y_true = bundle.y_test.to_numpy()

        pred_df = pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred": y_pred,
                "residual": y_true - y_pred,
            }
        )
        predictions[model_id] = pred_df

        row = {"model": model_id, "cv_score": best_score}
        row.update(regression_metrics(y_true=y_true, y_pred=y_pred))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(by="cv_score", ascending=False).reset_index(drop=True)

    result = TuningResult(
        phase=PHASE_TUNING,
        best_params=best_params,
        summary=summary_df,
        predictions=predictions,
        artifacts=[],
    )

    if output_cfg.output_dir is not None:
        from pathlib import Path

        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if output_cfg.export_excel:
            excel_path = output_dir / "tuning.xlsx"
            result.artifacts.append(write_tuning_excel(result, excel_path))

        if output_cfg.save_json:
            params_path = output_dir / "best_params.json"
            result.artifacts.append(write_json(result.best_params, params_path))

        if output_cfg.export_plots:
            for model_id, pred_df in predictions.items():
                plot_path = output_dir / f"tuning_{model_id}.png"
                result.artifacts.append(
                    save_interval_plot(pred_df, plot_path, title=f"Tuning Predictions - {model_id}")
                )

    return result
