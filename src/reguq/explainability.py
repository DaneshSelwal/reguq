"""Model explainability methods (SHAP, LIME, InterpretML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import coerce_output_config
from .constants import PHASE_EXPLAINABILITY
from .data import prepare_data_bundle
from .params import resolve_model_params
from .types import OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def _to_numpy(arr):
    """Convert to numpy array safely."""
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    return np.asarray(arr)


# =============================================================================
# SHAP Explainer
# =============================================================================


def explain_shap(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str] | None = None,
    max_samples: int = 100,
) -> dict[str, Any]:
    """Generate SHAP explanations."""
    try:
        from interpret.blackbox import ShapKernel
        use_interpret = True
    except ImportError:
        use_interpret = False

    X_train_np = _to_numpy(X_train)
    X_test_np = _to_numpy(X_test)
    feature_names = feature_names or [f"feature_{i}" for i in range(X_train_np.shape[1])]

    if use_interpret:
        if len(X_train_np) > max_samples:
            idx = np.random.choice(len(X_train_np), max_samples, replace=False)
            background = X_train_np[idx]
        else:
            background = X_train_np
        
        explainer = ShapKernel(model.predict, background, feature_names=feature_names)
        explanation = explainer.explain_local(X_test_np)

        feature_importances_shap = {}
        num_instances = len(X_test_np)
        for idx in range(num_instances):
            data = explanation.data(idx)
            for name, score in zip(data['names'], data['scores']):
                for fn in feature_names:
                    if fn in name:
                        feature_importances_shap[fn] = feature_importances_shap.get(fn, 0.0) + abs(score)
                        break

        feature_importance = [feature_importances_shap.get(name, 0.0) / num_instances for name in feature_names]

        return {
            "shap_values": explanation,
            "expected_value": 0.0,
            "feature_importance": np.array(feature_importance),
            "feature_names": feature_names,
        }

    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")

    # Subsample background data
    if len(X_train_np) > max_samples:
        idx = np.random.choice(len(X_train_np), max_samples, replace=False)
        background = X_train_np[idx]
    else:
        background = X_train_np

    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_np)

    # Feature importance (mean absolute SHAP)
    feature_importance = np.abs(shap_values).mean(axis=0)

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_importance": feature_importance,
        "feature_names": feature_names,
    }


# =============================================================================
# LIME Explainer
# =============================================================================


def explain_lime(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str] | None = None,
    num_features: int = 10,
    num_samples: int = 5000,
) -> dict[str, Any]:
    """Generate LIME explanations."""
    try:
        from interpret.blackbox import LimeTabular
        use_interpret = True
    except ImportError:
        use_interpret = False

    X_train_np = _to_numpy(X_train)
    X_test_np = _to_numpy(X_test)
    feature_names = feature_names or [f"feature_{i}" for i in range(X_train_np.shape[1])]

    if use_interpret:
        explainer = LimeTabular(
            model.predict,
            data=X_train_np,
            feature_names=feature_names,
            mode='regression'
        )
        explanation = explainer.explain_local(X_test_np)

        feature_importances_lime = {}
        num_instances = len(X_test_np)
        for idx in range(num_instances):
            data = explanation.data(idx)
            for name, score in zip(data['names'], data['scores']):
                for fn in feature_names:
                    if fn in name:
                        feature_importances_lime[fn] = feature_importances_lime.get(fn, 0.0) + abs(score)
                        break

        feature_importance = {name: feature_importances_lime.get(name, 0.0) / num_instances for name in feature_names}

        return {
            "explanations": explanation,
            "feature_importance": feature_importance,
            "feature_names": feature_names,
        }

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise ImportError("LIME not installed. Install with: pip install lime")

    explainer = LimeTabularExplainer(
        X_train_np,
        feature_names=feature_names,
        mode="regression",
    )

    # Generate explanations for test samples
    explanations = []
    for i in range(min(len(X_test_np), 100)):  # Limit to 100 samples
        exp = explainer.explain_instance(
            X_test_np[i],
            model.predict,
            num_features=num_features,
            num_samples=num_samples,
        )
        explanations.append(exp)

    # Aggregate feature importance
    feature_weights = {name: [] for name in feature_names}
    for exp in explanations:
        for name, weight in exp.as_list():
            # Extract feature name (LIME adds conditions)
            for fn in feature_names:
                if fn in name:
                    feature_weights[fn].append(abs(weight))
                    break

    feature_importance = {
        name: np.mean(weights) if weights else 0.0
        for name, weights in feature_weights.items()
    }

    return {
        "explanations": explanations,
        "feature_importance": feature_importance,
        "feature_names": feature_names,
    }


# =============================================================================
# InterpretML Explainer
# =============================================================================


def explain_interpret(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Generate InterpretML explanations using EBM.

    Reference:
        Nori, H., et al. "InterpretML: A Unified Framework for Machine
        Learning Interpretability." arXiv 2019.
        https://arxiv.org/abs/1909.09223

    Args:
        model: A fitted sklearn-compatible model (used for comparison).
        X_train: Training features.
        X_test: Test features to explain.
        y_train: Training targets.
        feature_names: Names of features.

    Returns:
        Dictionary with InterpretML explanations.
    """
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        from interpret import show
    except ImportError:
        raise ImportError(
            "InterpretML not installed. Install with: pip install interpret"
        )

    X_train_np = _to_numpy(X_train)
    X_test_np = _to_numpy(X_test)
    y_train_np = _to_numpy(y_train)

    feature_names = feature_names or [f"feature_{i}" for i in range(X_train_np.shape[1])]

    # Train an EBM for interpretable predictions
    ebm = ExplainableBoostingRegressor(feature_names=feature_names)
    ebm.fit(X_train_np, y_train_np)

    # Get global explanation
    global_exp = ebm.explain_global()

    # Get local explanations
    local_exp = ebm.explain_local(X_test_np[:100])  # Limit to 100 samples

    # Feature importance from EBM
    feature_importance = dict(zip(feature_names, ebm.feature_importances_))

    return {
        "ebm_model": ebm,
        "global_explanation": global_exp,
        "local_explanation": local_exp,
        "feature_importance": feature_importance,
        "feature_names": feature_names,
        "ebm_predictions": ebm.predict(X_test_np),
    }


# =============================================================================
# Main Runner Function
# =============================================================================


def run_explainability(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
    methods: list[str] | None = None,
    shap_config: Mapping[str, Any] | None = None,
    lime_config: Mapping[str, Any] | None = None,
) -> PhaseResult:
    """Run explainability analysis on trained models.

    Supported methods:
    - shap: SHAP (SHapley Additive exPlanations)
    - lime: LIME (Local Interpretable Model-agnostic Explanations)
    - interpret: InterpretML (Explainable Boosting Machine)

    Args:
        data: Input data (DataFrame, CSV path, or dict with train/test).
        target_col: Name of the target column.
        models: List of model IDs to explain. Defaults to all supported models.
        params_source: Source for model parameters.
        output_config: Output configuration.
        split_config: Train/test split configuration.
        methods: List of methods to run (default: ["shap"]).
        shap_config: Configuration for SHAP (max_samples).
        lime_config: Configuration for LIME (num_features, num_samples).

    Returns:
        PhaseResult with explanations and feature importance.
    """
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_EXPLAINABILITY)
    output_cfg = coerce_output_config(output_config)

    methods = methods or ["shap"]
    shap_cfg = dict(shap_config or {})
    lime_cfg = dict(lime_config or {})

    model_params, tuned_params = resolve_model_params(
        models=model_ids,
        params_source=params_source,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
    )

    feature_names = list(bundle.X_train.columns)
    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}
    explanations: dict[str, dict[str, Any]] = {}

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))
        estimator = model_registry.build_estimator(
            model_id=model_id, phase=PHASE_EXPLAINABILITY, params=params
        )
        estimator.fit(bundle.X_train, bundle.y_train)

        for method in methods:
            key = f"{model_id}_{method}"

            try:
                if method == "shap":
                    result = explain_shap(
                        model=estimator,
                        X_train=bundle.X_train,
                        X_test=bundle.X_test,
                        feature_names=feature_names,
                        max_samples=shap_cfg.get("max_samples", 100),
                    )
                    feature_importance = dict(
                        zip(result["feature_names"], result["feature_importance"])
                    )

                elif method == "lime":
                    result = explain_lime(
                        model=estimator,
                        X_train=bundle.X_train,
                        X_test=bundle.X_test,
                        feature_names=feature_names,
                        num_features=lime_cfg.get("num_features", 10),
                        num_samples=lime_cfg.get("num_samples", 5000),
                    )
                    feature_importance = result["feature_importance"]

                elif method == "interpret":
                    result = explain_interpret(
                        model=estimator,
                        X_train=bundle.X_train,
                        X_test=bundle.X_test,
                        y_train=bundle.y_train,
                        feature_names=feature_names,
                    )
                    feature_importance = result["feature_importance"]

                else:
                    raise ValueError(f"Unknown explainability method '{method}'")

                explanations[key] = result

                # Create feature importance DataFrame
                importance_df = pd.DataFrame(
                    {
                        "feature": list(feature_importance.keys()),
                        "importance": list(feature_importance.values()),
                    }
                ).sort_values("importance", ascending=False)

                predictions[key] = importance_df

                # Summary metrics
                row = {
                    "model": model_id,
                    "method": method,
                    "top_feature": importance_df.iloc[0]["feature"],
                    "top_importance": importance_df.iloc[0]["importance"],
                    "num_features": len(importance_df),
                }
                metrics_rows.append(row)

            except ImportError as e:
                continue
            except Exception as e:
                continue

    metrics_df = pd.DataFrame(metrics_rows)

    result = PhaseResult(
        phase=PHASE_EXPLAINABILITY,
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )

    # Export results
    if output_cfg.output_dir is not None:
        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt
        import logging

        # Save feature importance plots
        side_by_side_plots = {}
        single_plots = {}

        # 1. Generate side-by-side plots for models having both shap and lime
        for model_id in model_ids:
            key_shap = f"{model_id}_shap"
            key_lime = f"{model_id}_lime"
            if key_shap in predictions and key_lime in predictions:
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    df_shap = predictions[key_shap]
                    top_n = min(10, len(df_shap))
                    df_shap_top = df_shap.head(top_n)
                    ax1.bar(df_shap_top["feature"], df_shap_top["importance"], color='skyblue')
                    ax1.set_ylabel("SHAP Importance")
                    ax1.set_title("SHAP Feature Importance")
                    ax1.tick_params(axis='x', rotation=45)

                    df_lime = predictions[key_lime]
                    df_lime_top = df_lime.head(top_n)
                    ax2.bar(df_lime_top["feature"], df_lime_top["importance"], color='salmon')
                    ax2.set_ylabel("LIME Importance")
                    ax2.set_title("LIME Feature Importance")
                    ax2.tick_params(axis='x', rotation=45)

                    plt.tight_layout()
                    plot_path = output_dir / f"feature_importance_{model_id}_side_by_side.png"
                    fig.savefig(plot_path, dpi=150)
                    plt.close(fig)

                    side_by_side_plots[model_id] = plot_path
                    result.artifacts.append(plot_path)
                except Exception as e:
                    logging.warning(f"Failed to generate side-by-side plot for {model_id}: {e}")

        # 2. Generate single plots for anything else
        for key, df in predictions.items():
            model_id = key.split("_")[0]
            if model_id in side_by_side_plots:
                continue
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_n = min(15, len(df))
                df_top = df.head(top_n)
                ax.bar(df_top["feature"], df_top["importance"], color='skyblue')
                ax.set_ylabel("Importance")
                ax.set_xlabel("Feature")
                ax.set_title(f"Feature Importance - {key}")
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()

                plot_path = output_dir / f"feature_importance_{key}.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)

                single_plots[key] = plot_path
                result.artifacts.append(plot_path)
            except Exception as e:
                logging.warning(f"Failed to generate plot for {key}: {e}")

        # Save importance data to Excel
        if output_cfg.export_excel:
            try:
                excel_path = output_dir / "explainability.xlsx"
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    metrics_df.to_excel(writer, sheet_name="Summary", index=False)
                    for key, df in predictions.items():
                        sheet_name = key[:31]  # Excel sheet name limit
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Embed explanation plots
                if output_cfg.embed_excel_charts:
                    images_by_sheet = {}
                    for key in predictions.keys():
                        model_id = key.split("_")[0]
                        sheet_name = key[:31]
                        if model_id in side_by_side_plots:
                            images_by_sheet[sheet_name] = [side_by_side_plots[model_id]]
                        elif key in single_plots:
                            images_by_sheet[sheet_name] = [single_plots[key]]

                    if images_by_sheet:
                        from .export import embed_images_in_excel
                        embed_images_in_excel(workbook_path=excel_path, images_by_sheet=images_by_sheet)

                result.artifacts.append(excel_path)
            except Exception as e:
                logging.warning(f"Failed to write Excel: {e}")

    return result
