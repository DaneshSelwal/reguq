from __future__ import annotations

import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from reguq.quantile import run_quantile
from reguq.probabilistic import run_probabilistic
from reguq.conformal_standard import run_conformal_standard
from reguq.conformal_advanced import run_conformal_advanced
from reguq.tuning import run_tuning
from reguq.explainability import run_explainability
from reguq.probabilistic_advanced import run_probabilistic_advanced
from reguq.constants import PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_TUNING
from reguq.types import OutputConfig
try:
    from .conftest import is_package_installed
except (ImportError, ValueError):
    def is_package_installed(package_name: str) -> bool:
        import importlib
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False

# Helper to select a model for testing
def get_test_model(available_models):
    for m in ["lightgbm", "xgboost", "catboost", "randomforest", "gradientboosting"]:
        if m in available_models:
            return m
    return "gradientboosting"

# =============================================================================
# Core Feature 1: Quantile Regression
# =============================================================================

def test_quantile_regression_basic(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    assert model in result.predictions
    pred_df = result.predictions[model]
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns
    assert len(pred_df) == len(test_df)

def test_quantile_regression_custom_quantiles(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        quantiles=(0.1, 0.9),
    )
    
    pred_df = result.predictions[model]
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])

def test_quantile_regression_metrics(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    assert not result.metrics.empty
    metrics = result.metrics.iloc[0].to_dict()
    assert "coverage" in metrics
    assert "avg_interval_width" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics

def test_quantile_regression_multiple_models(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    models = [m for m in ["randomforest", "gradientboosting"] if m in available_models]
    if not models:
        models = [available_models[0]]
        
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=models,
        params_source={"mode": "load_or_tune", "params": {m: fast_params.get(m, {}) for m in models}},
    )
    
    for model in models:
        assert model in result.predictions
        assert len(result.predictions[model]) == len(test_df)

def test_quantile_regression_export_excel(e2e_synthetic_data, available_models, fast_params, tmp_path):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    output_dir = tmp_path / "quantile_outputs"
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        output_config=OutputConfig(
            output_dir=str(output_dir),
            export_excel=True,
            export_plots=False,
            save_json=True,
        ),
    )
    
    excel_file = output_dir / "quantile.xlsx"
    assert excel_file.exists()
    assert any(str(excel_file) in str(art) for art in result.artifacts)


# =============================================================================
# Core Feature 2: Probabilistic Models
# =============================================================================

def test_probabilistic_basic(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = "ngboost" if "ngboost" in available_models else get_test_model(available_models)
    
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    assert model in result.predictions
    pred_df = result.predictions[model]
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert "y_std" in pred_df.columns
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns

def test_probabilistic_custom_alpha(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = "ngboost" if "ngboost" in available_models else get_test_model(available_models)
    
    res_90 = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        alpha=0.10,
    )
    
    res_95 = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        alpha=0.05,
    )
    
    width_90 = (res_90.predictions[model]["y_upper"] - res_90.predictions[model]["y_lower"]).mean()
    width_95 = (res_95.predictions[model]["y_upper"] - res_95.predictions[model]["y_lower"]).mean()
    assert width_95 > width_90

def test_probabilistic_metrics(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = "ngboost" if "ngboost" in available_models else get_test_model(available_models)
    
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    metrics = result.metrics.iloc[0].to_dict()
    assert "nll" in metrics
    assert "crps" in metrics

def test_probabilistic_multiple_models(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    models = [m for m in ["randomforest", "gradientboosting"] if m in available_models]
    if not models:
        models = [available_models[0]]
        
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=models,
        params_source={"mode": "load_or_tune", "params": {m: fast_params.get(m, {}) for m in models}},
    )
    
    for m in models:
        assert m in result.predictions

def test_probabilistic_export(e2e_synthetic_data, available_models, fast_params, tmp_path):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    output_dir = tmp_path / "prob_outputs"
    
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        output_config={
            "output_dir": str(output_dir),
            "export_excel": True,
        },
    )
    
    excel_file = output_dir / "probabilistic.xlsx"
    assert excel_file.exists()


# =============================================================================
# Core Feature 3: Conformal Prediction
# =============================================================================

def test_conformal_standard_mapie(e2e_synthetic_data, available_models, fast_params):
    if not is_package_installed("mapie"):
        pytest.skip("MAPIE is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["mapie"]},
    )
    
    assert "mapie" in result.methods
    pred_df = result.methods["mapie"].predictions[model]
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns

def test_conformal_standard_puncc(e2e_synthetic_data, available_models, fast_params):
    if not is_package_installed("puncc"):
        pytest.skip("PUNCC is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["puncc"]},
    )
    
    assert "puncc" in result.methods
    pred_df = result.methods["puncc"].predictions[model]
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns

def test_conformal_advanced_cvplus(e2e_synthetic_data, available_models, fast_params):
    if not is_package_installed("puncc"):
        pytest.skip("PUNCC is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["cvplus"]},
    )
    
    assert "cvplus" in result.methods
    pred_df = result.methods["cvplus"].predictions[model]
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns

def test_conformal_advanced_cqr(e2e_synthetic_data, available_models, fast_params):
    if not is_package_installed("puncc"):
        pytest.skip("PUNCC is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # CQR needs quantile estimator
    result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["cqr"]},
    )
    
    assert "cqr" in result.methods
    pred_df = result.methods["cqr"].predictions[model]
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns

def test_conformal_advanced_online(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["online_split"]},
    )
    
    assert "online_split" in result.methods
    pred_df = result.methods["online_split"].predictions[model]
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns


# =============================================================================
# Core Feature 4: Hyperparameter Tuning
# =============================================================================

def test_tuning_optuna(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 2, "cv": 2, "random_state": 42},
    )
    
    assert model in result.best_params
    assert isinstance(result.best_params[model], dict)
    assert not result.summary.empty

def test_tuning_custom_scorer(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 2, "cv": 2, "scoring": "neg_mean_absolute_error", "random_state": 42},
    )
    
    assert model in result.best_params

def test_tuning_multiple_models(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    models = [m for m in ["randomforest", "gradientboosting"] if m in available_models]
    if not models:
        models = [available_models[0]]
        
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=models,
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    
    for model in models:
        assert model in result.best_params

def test_tuning_export(e2e_synthetic_data, available_models, tmp_path):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    output_dir = tmp_path / "tuning_outputs"
    
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
        output_config={
            "output_dir": str(output_dir),
            "save_json": True,
        },
    )
    
    assert (output_dir / "best_params.json").exists()

def test_tuning_summary_metrics(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    
    assert "cv_score" in result.summary.columns


# =============================================================================
# Core Feature 5: Explainability
# =============================================================================

def is_explain_method_available(method: str) -> bool:
    if method == "shap":
        return is_package_installed("shap")
    if method == "lime":
        return is_package_installed("lime")
    if method == "interpret":
        return is_package_installed("interpret")
    return False

def test_explainability_shap(e2e_synthetic_data, available_models, fast_params):
    if not is_explain_method_available("shap"):
        pytest.skip("SHAP not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["shap"],
        shap_config={"max_samples": 20},
    )
    
    key = f"{model}_shap"
    assert key in result.predictions
    assert "feature" in result.predictions[key].columns
    assert "importance" in result.predictions[key].columns

def test_explainability_lime(e2e_synthetic_data, available_models, fast_params):
    if not is_explain_method_available("lime"):
        pytest.skip("LIME not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["lime"],
        lime_config={"num_features": 2, "num_samples": 10},
    )
    
    key = f"{model}_lime"
    assert key in result.predictions
    assert "feature" in result.predictions[key].columns
    assert "importance" in result.predictions[key].columns

def test_explainability_interpret(e2e_synthetic_data, available_models, fast_params):
    if not is_explain_method_available("interpret"):
        pytest.skip("InterpretML not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["interpret"],
    )
    
    key = f"{model}_interpret"
    assert key in result.predictions
    assert "feature" in result.predictions[key].columns
    assert "importance" in result.predictions[key].columns

def test_explainability_plots(e2e_synthetic_data, available_models, fast_params, tmp_path):
    # Skip if no explainability package is installed
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    output_dir = tmp_path / "explain_plots"
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
        output_config={
            "output_dir": str(output_dir),
            "export_plots": True,
        },
    )
    
    # Check if plot was saved
    pngs = list(output_dir.glob("*.png"))
    assert len(pngs) > 0

def test_explainability_multiple_models(e2e_synthetic_data, available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df, test_df = e2e_synthetic_data
    models = [m for m in ["randomforest", "gradientboosting"] if m in available_models]
    if not models:
        models = [available_models[0]]
        
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=models,
        params_source={"mode": "load_or_tune", "params": {m: fast_params.get(m, {}) for m in models}},
        methods=[available_methods[0]],
    )
    
    for model in models:
        key = f"{model}_{available_methods[0]}"
        assert key in result.predictions


def test_probabilistic_advanced_basic(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_probabilistic_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["ibug", "hcm"],
        hcm_config={"epochs": 2},
    )
    
    assert not result.metrics.empty
    for method in ["ibug", "hcm"]:
        key = f"{model}_{method}"
        assert key in result.predictions
        pred_df = result.predictions[key]
        assert "y_true" in pred_df.columns
        assert "y_pred" in pred_df.columns
        assert "y_std" in pred_df.columns
        assert "y_lower" in pred_df.columns
        assert "y_upper" in pred_df.columns

def test_probabilistic_advanced_card(e2e_synthetic_data, available_models, fast_params):
    if not is_package_installed("torch"):
        pytest.skip("PyTorch is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_probabilistic_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["card"],
        card_config={"epochs": 2},
    )
    
    key = f"{model}_card"
    assert key in result.predictions
    pred_df = result.predictions[key]
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert "y_std" in pred_df.columns
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns

def test_probabilistic_advanced_treeffuser(e2e_synthetic_data, available_models, fast_params):
    if not is_package_installed("treeffuser"):
        pytest.skip("treeffuser is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_probabilistic_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["treeffuser"],
    )
    
    key = f"{model}_treeffuser"
    assert key in result.predictions
    pred_df = result.predictions[key]
    assert "y_true" in pred_df.columns
    assert "y_pred" in pred_df.columns
    assert "y_std" in pred_df.columns
    assert "y_lower" in pred_df.columns
    assert "y_upper" in pred_df.columns
