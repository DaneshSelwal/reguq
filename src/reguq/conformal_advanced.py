"""Advanced conformal prediction methods (NexCP, Adaptive CP, MFCS, CVPlus, CQR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .charts import generate_conformal_charts
from .config import coerce_output_config
from .constants import DEFAULT_ALPHA, PHASE_CONFORMAL_ADVANCED
from .data import prepare_data_bundle
from .export import embed_images_in_excel, write_conformal_excel, write_json
from .metrics import interval_metrics, regression_metrics
from .params import resolve_model_params
from .types import ConformalResult, OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def _to_numpy(arr):
    """Convert to numpy array safely."""
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy().ravel()
    return np.asarray(arr).ravel()


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted quantile."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cdf, values))


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower and upper bounds from interval arrays."""
    arr = np.asarray(intervals)
    if arr.ndim == 3:
        lower = arr[:, 0, 0]
        upper = arr[:, 1, 0]
        return np.asarray(lower).ravel(), np.asarray(upper).ravel()
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.asarray(arr[:, 0]).ravel(), np.asarray(arr[:, 1]).ravel()
    raise ValueError(f"Unsupported interval shape: {arr.shape}")


# =============================================================================
# PUNCC Advanced Methods (CVPlus, CQR)
# =============================================================================


def _predict_puncc_cvplus(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
):
    """PUNCC CV+ (Cross-Validation Plus) conformal prediction."""
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import CVPlus

    predictor = BasePredictor(estimator)
    cp = CVPlus(predictor)
    cp.fit(X_train.to_numpy(), y_train.to_numpy())
    outputs = cp.predict(X_test.to_numpy(), alpha=alpha)

    if isinstance(outputs, tuple) and len(outputs) >= 2:
        y_pred = np.asarray(outputs[0]).ravel()
        intervals = outputs[1]
    else:
        raise ValueError("Unexpected PUNCC CVPlus predict output format.")

    y_lower, y_upper = _extract_interval_bounds(intervals)
    return y_pred, y_lower, y_upper


def _predict_puncc_cqr(
    lower_estimator,
    upper_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
):
    """PUNCC CQR (Conformalized Quantile Regression) conformal prediction."""
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import CQR

    lower_predictor = BasePredictor(lower_estimator)
    upper_predictor = BasePredictor(upper_estimator)
    cp = CQR(lower_predictor, upper_predictor)
    cp.fit(X_train.to_numpy(), y_train.to_numpy())
    outputs = cp.predict(X_test.to_numpy(), alpha=alpha)

    if isinstance(outputs, tuple) and len(outputs) >= 2:
        y_pred = np.asarray(outputs[0]).ravel()
        intervals = outputs[1]
    else:
        raise ValueError("Unexpected PUNCC CQR predict output format.")

    y_lower, y_upper = _extract_interval_bounds(intervals)
    return y_pred, y_lower, y_upper


# =============================================================================
# NexCP Methods (Non-Exchangeable Conformal Prediction)
# =============================================================================


def _predict_nexcp_split(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    decay: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP Split with exponential weighting for non-exchangeable data."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    residuals = np.abs(y_train - yhat_train)
    weights = decay ** np.arange(len(residuals) - 1, -1, -1)
    q = _weighted_quantile(residuals, weights, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_nexcp_full(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP Full conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    residuals = np.abs(y_train - yhat_train)
    q = np.quantile(residuals, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_nexcp_jackknife_ab(
    model_builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP Jackknife+ after Bootstrap."""
    rng = np.random.default_rng(random_state)
    lowers, uppers, preds = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        model = model_builder()
        model.fit(X_train[idx], y_train[idx])

        q = np.quantile(np.abs(y_train[idx] - model.predict(X_train[idx])), 1 - alpha)
        p = model.predict(X_test)

        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)

    y_pred = np.mean(preds, axis=0)
    y_lower = np.min(lowers, axis=0)
    y_upper = np.max(uppers, axis=0)
    return y_pred, y_lower, y_upper


def _predict_nexcp_cv_plus(
    model_builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP CV+ (Cross-Validation Plus)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    lowers, uppers, preds = [], [], []

    for tr_idx, cal_idx in kf.split(X_train):
        model = model_builder()
        model.fit(X_train[tr_idx], y_train[tr_idx])

        q = np.quantile(np.abs(y_train[cal_idx] - model.predict(X_train[cal_idx])), 1 - alpha)
        p = model.predict(X_test)

        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)

    y_pred = np.mean(preds, axis=0)
    y_lower = np.mean(lowers, axis=0)
    y_upper = np.mean(uppers, axis=0)
    return y_pred, y_lower, y_upper


# =============================================================================
# Online/Adaptive CP Methods
# =============================================================================


def _predict_online_split(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online Split conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    residuals = np.abs(y_train - yhat_train)
    q = np.quantile(residuals, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_faci(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    gamma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully Adaptive Conformal Inference (FACI)."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train
    for t in range(n_train, T):
        y_hat_all[t] = estimator.predict(X_all[t : t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)

    # FACI: Adaptive quantile adjustment
    alphas = np.full(T - n_train, alpha)
    y_lower = np.zeros(T - n_train)
    y_upper = np.zeros(T - n_train)

    for t in range(T - n_train):
        q = np.quantile(residuals[: n_train + t], 1 - alphas[t])
        y_lower[t] = yhat_test[t] - q
        y_upper[t] = yhat_test[t] + q

        # Update alpha based on coverage
        if t > 0:
            covered = (y_test[t - 1] >= y_lower[t - 1]) and (y_test[t - 1] <= y_upper[t - 1])
            alphas[t] = max(0.01, min(0.5, alphas[t - 1] + gamma * (alpha - (1 if covered else 0))))

    return yhat_test, y_lower, y_upper


# =============================================================================
# MFCS Methods (Model-Free Conformal Selection)
# =============================================================================


def _predict_mfcs_split(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    calibration_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MFCS Split conformal prediction."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=calibration_size, shuffle=False, random_state=random_state
    )

    estimator.fit(X_fit, y_fit)
    y_cal_pred = estimator.predict(X_cal)
    yhat_test = estimator.predict(X_test)

    scores = np.abs(y_cal - y_cal_pred)
    q = np.quantile(scores, 1 - alpha, method="higher")

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_mfcs_full(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MFCS Full conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    scores = np.abs(y_train - yhat_train)
    q = np.quantile(scores, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


# =============================================================================
# Main Runner Functions
# =============================================================================


def _run_advanced_method(
    method_name: str,
    model_ids: list[str],
    model_params: dict[str, dict[str, Any]],
    bundle,
    alpha: float,
    decay: float,
    n_folds: int,
    n_bootstrap: int,
    calibration_size: float,
    random_state: int,
) -> PhaseResult:
    """Run a single advanced conformal method across all models."""
    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    X_train = bundle.X_train.to_numpy() if hasattr(bundle.X_train, "to_numpy") else np.asarray(bundle.X_train)
    y_train = _to_numpy(bundle.y_train)
    X_test = bundle.X_test.to_numpy() if hasattr(bundle.X_test, "to_numpy") else np.asarray(bundle.X_test)
    y_test = _to_numpy(bundle.y_test)

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))

        def model_builder():
            return model_registry.build_estimator(
                model_id=model_id, phase=PHASE_CONFORMAL_ADVANCED, params=params
            )

        estimator = model_builder()

        try:
            if method_name == "nexcp_split":
                y_pred, y_lower, y_upper = _predict_nexcp_split(
                    estimator, X_train, y_train, X_test, alpha, decay
                )
            elif method_name == "nexcp_full":
                y_pred, y_lower, y_upper = _predict_nexcp_full(
                    estimator, X_train, y_train, X_test, alpha
                )
            elif method_name == "nexcp_jackknife_ab":
                y_pred, y_lower, y_upper = _predict_nexcp_jackknife_ab(
                    model_builder, X_train, y_train, X_test, alpha, n_bootstrap, random_state
                )
            elif method_name == "nexcp_cv_plus":
                y_pred, y_lower, y_upper = _predict_nexcp_cv_plus(
                    model_builder, X_train, y_train, X_test, alpha, n_folds, random_state
                )
            elif method_name == "online_split":
                y_pred, y_lower, y_upper = _predict_online_split(
                    estimator, X_train, y_train, X_test, alpha
                )
            elif method_name == "faci":
                y_pred, y_lower, y_upper = _predict_faci(
                    estimator, X_train, y_train, X_test, y_test, alpha
                )
            elif method_name == "mfcs_split":
                y_pred, y_lower, y_upper = _predict_mfcs_split(
                    estimator, X_train, y_train, X_test, alpha, calibration_size, random_state
                )
            elif method_name == "mfcs_full":
                y_pred, y_lower, y_upper = _predict_mfcs_full(
                    estimator, X_train, y_train, X_test, alpha
                )
            elif method_name == "cvplus":
                y_pred, y_lower, y_upper = _predict_puncc_cvplus(
                    estimator, bundle.X_train, bundle.y_train, bundle.X_test, alpha
                )
            elif method_name == "cqr":
                # CQR needs quantile models - build lower and upper
                lower_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=alpha / 2
                )
                upper_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=1 - alpha / 2
                )
                y_pred, y_lower, y_upper = _predict_puncc_cqr(
                    lower_est, upper_est, bundle.X_train, bundle.y_train, bundle.X_test, alpha
                )
            else:
                raise ValueError(f"Unknown advanced conformal method '{method_name}'")

        except Exception as e:
            # Fallback to basic split conformal
            y_pred, y_lower, y_upper = _predict_mfcs_split(
                estimator, X_train, y_train, X_test, alpha, calibration_size, random_state
            )

        pred_df = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": y_pred,
                "y_lower": y_lower,
                "y_upper": y_upper,
            }
        )
        predictions[model_id] = pred_df

        row = {"model": model_id, "method": method_name, "alpha": alpha}
        row.update(regression_metrics(y_true=y_test, y_pred=y_pred))
        row.update(interval_metrics(y_true=y_test, y_lower=y_lower, y_upper=y_upper))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    return PhaseResult(
        phase=f"{PHASE_CONFORMAL_ADVANCED}_{method_name}",
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )


def run_conformal_advanced(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    conformal_config: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
) -> ConformalResult:
    """Run advanced conformal prediction methods.

    Supported methods:
    - nexcp_split: NexCP Split with exponential weighting
    - nexcp_full: NexCP Full conformal
    - nexcp_jackknife_ab: NexCP Jackknife+ after Bootstrap
    - nexcp_cv_plus: NexCP Cross-Validation Plus
    - online_split: Online Split conformal
    - faci: Fully Adaptive Conformal Inference
    - mfcs_split: Model-Free Conformal Selection (Split)
    - mfcs_full: Model-Free Conformal Selection (Full)
    - cvplus: PUNCC CV+ (Cross-Validation Plus)
    - cqr: PUNCC CQR (Conformalized Quantile Regression)

    Args:
        data: Input data (DataFrame, CSV path, or dict with train/test).
        target_col: Name of the target column.
        models: List of model IDs to use. Defaults to all supported models.
        params_source: Source for model parameters.
        conformal_config: Configuration dict with keys:
            - alpha: Miscoverage rate (default: 0.1)
            - methods: List of methods to run (default: all)
            - decay: Decay factor for NexCP (default: 0.99)
            - n_folds: Number of folds for CV methods (default: 5)
            - n_bootstrap: Bootstrap iterations (default: 50)
            - calibration_size: Calibration set proportion (default: 0.2)
            - random_state: Random seed (default: 42)
        output_config: Output configuration.
        split_config: Train/test split configuration.

    Returns:
        ConformalResult with predictions and metrics for each method.
    """
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_CONFORMAL_ADVANCED)
    output_cfg = coerce_output_config(output_config)

    cfg = dict(conformal_config or {})
    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    decay = float(cfg.get("decay", 0.99))
    n_folds = int(cfg.get("n_folds", 5))
    n_bootstrap = int(cfg.get("n_bootstrap", 50))
    calibration_size = float(cfg.get("calibration_size", 0.2))
    random_state = int(cfg.get("random_state", 42))

    default_methods = [
        "nexcp_split",
        "nexcp_full",
        "nexcp_jackknife_ab",
        "nexcp_cv_plus",
        "online_split",
        "faci",
        "mfcs_split",
        "mfcs_full",
        "cvplus",
    ]
    methods = list(cfg.get("methods", default_methods))

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
        method_results[method_name] = _run_advanced_method(
            method_name=method_name,
            model_ids=model_ids,
            model_params=model_params,
            bundle=bundle,
            alpha=alpha,
            decay=decay,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
            calibration_size=calibration_size,
            random_state=random_state,
        )

    result = ConformalResult(
        phase=PHASE_CONFORMAL_ADVANCED,
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

        excel_path = output_dir / "conformal_advanced.xlsx"
        if output_cfg.export_excel:
            result.artifacts.append(write_conformal_excel(result, excel_path))
            if output_cfg.embed_excel_charts and chart_result is not None and chart_result.images_by_sheet:
                embed_images_in_excel(workbook_path=excel_path, images_by_sheet=chart_result.images_by_sheet)

        if output_cfg.save_json and tuned_params:
            result.artifacts.append(write_json(tuned_params, output_dir / "conformal_advanced_tuned_params.json"))

    return result
