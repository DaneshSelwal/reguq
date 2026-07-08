from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from reguq.quantile import run_quantile
from reguq.probabilistic import run_probabilistic
from reguq.conformal_standard import run_conformal_standard
from reguq.conformal_advanced import run_conformal_advanced
from reguq.tuning import run_tuning
from reguq.explainability import run_explainability
from reguq.runner import run_from_config
from reguq.types import OutputConfig

# Helper to select a model for testing
def get_test_model(available_models):
    for m in ["lightgbm", "xgboost", "catboost", "randomforest", "gradientboosting"]:
        if m in available_models:
            return m
    return "gradientboosting"

def is_explain_method_available(method: str) -> bool:
    import importlib
    try:
        importlib.import_module(method)
        return True
    except ImportError:
        return False

# =============================================================================
# Combo 1: Tuning -> Quantile Regression
# =============================================================================
def test_combo_tuning_then_quantile(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # 1. Tune
    tuning_result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    
    best_params = tuning_result.best_params
    
    # 2. Run quantile using the best params
    quantile_result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": best_params},
    )
    
    assert model in quantile_result.predictions
    assert quantile_result.params[model] == best_params[model]


# =============================================================================
# Combo 2: Tuning -> Probabilistic Regression
# =============================================================================
def test_combo_tuning_then_probabilistic(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # 1. Tune
    tuning_result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    
    best_params = tuning_result.best_params
    
    # 2. Run probabilistic using best params
    prob_result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": best_params},
    )
    
    assert model in prob_result.predictions
    assert prob_result.params[model] == best_params[model]


# =============================================================================
# Combo 3: Tuning -> Conformal Prediction
# =============================================================================
def test_combo_tuning_then_conformal(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # 1. Tune
    tuning_result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    
    best_params = tuning_result.best_params
    
    # 2. Run conformal using best params
    conformal_result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": best_params},
        conformal_config={"alpha": 0.1, "methods": ["mapie", "puncc"]},
    )
    
    for method in conformal_result.methods:
        assert model in conformal_result.methods[method].predictions
        assert conformal_result.methods[method].params[model] == best_params[model]


# =============================================================================
# Combo 4: Quantile Regression -> Conformal Prediction (CQR)
# =============================================================================
def test_combo_quantile_then_conformal_cqr(e2e_synthetic_data, available_models, fast_params):
    if not is_explain_method_available("deel.puncc"):
        pytest.skip("PUNCC is not installed")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # Run advanced conformal prediction with CQR which builds two quantile estimators
    conformal_result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["cqr"]},
    )
    
    assert "cqr" in conformal_result.methods
    assert model in conformal_result.methods["cqr"].predictions


# =============================================================================
# Combo 5: Probabilistic Regression -> Explainability
# =============================================================================
def test_combo_probabilistic_then_explain(e2e_synthetic_data, available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # 1. Probabilistic predictions
    prob_result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    # 2. Explain the model used
    explain_result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
    )
    
    key = f"{model}_{available_methods[0]}"
    assert key in explain_result.predictions


# =============================================================================
# Combo 6: Conformal Prediction (Advanced) -> Explainability
# =============================================================================
def test_combo_conformal_advanced_then_explain(e2e_synthetic_data, available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # 1. Advanced conformal
    conformal_result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["nexcp_split"]},
    )
    
    # 2. Explain
    explain_result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
    )
    
    key = f"{model}_{available_methods[0]}"
    assert key in explain_result.predictions


# =============================================================================
# Combo 7: End-to-End Pipeline config execution
# =============================================================================
def test_combo_pipeline_config_runner(e2e_synthetic_paths, available_models, tmp_path):
    train_path, test_path = e2e_synthetic_paths
    model = get_test_model(available_models)
    output_dir = tmp_path / "pipeline_outputs"
    
    config = {
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_col": "target",
        },
        "models": [model],
        "phases": ["quantile", "probabilistic", "conformal_standard"],
        "params_source": {"mode": "defaults"},
        "output": {
            "output_dir": str(output_dir),
            "export_excel": True,
            "save_json": True,
        },
    }
    
    run_result = run_from_config(config)
    
    assert run_result.output_dir == output_dir
    assert "quantile" in run_result.results
    assert "probabilistic" in run_result.results
    assert "conformal_standard" in run_result.results
    
    assert (output_dir / "quantile.xlsx").exists()
    assert (output_dir / "probabilistic.xlsx").exists()
    assert (output_dir / "conformal_standard.xlsx").exists()
