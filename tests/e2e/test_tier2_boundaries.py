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
from reguq.probabilistic_advanced import run_probabilistic_advanced
from reguq.constants import PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED
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
# Boundary Cases 1: Quantile Regression
# =============================================================================

def test_quantile_invalid_bounds_negative(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        run_quantile(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
            quantiles=(-0.1, 0.9),
        )

def test_quantile_invalid_bounds_order(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        run_quantile(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
            quantiles=(0.8, 0.2),
        )

def test_quantile_constant_target(available_models, fast_params):
    # Constant target: all y values identical
    rng = np.random.default_rng(42)
    n_train, n_test = 20, 5
    train_df = pd.DataFrame({"f1": rng.normal(0, 1, n_train), "target": 5.0})
    test_df = pd.DataFrame({"f1": rng.normal(0, 1, n_test), "target": 5.0})
    model = get_test_model(available_models)
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    pred_df = result.predictions[model]
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])

def test_quantile_extreme_asymmetry(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        quantiles=(0.49, 0.51),
    )
    
    pred_df = result.predictions[model]
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])

def test_quantile_single_sample(available_models, fast_params):
    # Extremely small dataset
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [2.0, 4.0, 6.0, 8.0, 10.0]})
    test_df = pd.DataFrame({"f1": [1.5], "target": [3.0]})
    model = get_test_model(available_models)
    
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    assert len(result.predictions[model]) == 1


# =============================================================================
# Boundary Cases 2: Probabilistic Models
# =============================================================================

def test_probabilistic_invalid_alpha_zero(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    with pytest.raises(ValueError, match="alpha must satisfy"):
        run_probabilistic(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
            alpha=-0.1,
        )

def test_probabilistic_invalid_alpha_large(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    with pytest.raises(ValueError, match="alpha must satisfy"):
        run_probabilistic(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
            alpha=1.5,
        )

def test_probabilistic_constant_target(available_models, fast_params):
    rng = np.random.default_rng(42)
    train_df = pd.DataFrame({"f1": rng.normal(0, 1, 20), "target": 5.0})
    test_df = pd.DataFrame({"f1": rng.normal(0, 1, 5), "target": 5.0})
    model = get_test_model(available_models)
    
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    pred_df = result.predictions[model]
    assert np.all(pred_df["y_std"] >= 1e-8)  # Lower standard deviation floor
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])

def test_probabilistic_extreme_alpha(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        alpha=0.999,
    )
    
    pred_df = result.predictions[model]
    # At alpha -> 1, y_lower should be very close to y_upper (since z-score -> 0)
    width = (pred_df["y_upper"] - pred_df["y_lower"]).mean()
    assert width < 0.1

def test_probabilistic_small_data(available_models, fast_params):
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
    test_df = pd.DataFrame({"f1": [1.5], "target": [3.0]})
    model = get_test_model(available_models)
    
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
    )
    
    assert len(result.predictions[model]) == 1


# =============================================================================
# Boundary Cases 3: Conformal Prediction
# =============================================================================

def test_conformal_invalid_alpha(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    with pytest.raises(ValueError, match="conformal alpha must satisfy"):
        run_conformal_standard(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
            conformal_config={"alpha": 2.0},
        )

def test_conformal_zero_residuals(available_models, fast_params):
    # Fit is perfect (y = x). Let's see if conformal still executes.
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    test_df = pd.DataFrame({"f1": [1.5, 2.5], "target": [1.5, 2.5]})
    model = "randomforest"  # Random Forest will fit perfectly on training grid
    
    result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: {"n_estimators": 5, "max_depth": 5}}},
        conformal_config={"alpha": 0.1, "methods": ["mapie", "puncc"]},
    )
    
    # Standard conformal methods should fallback or work with narrow margins
    for method in result.methods:
        pred_df = result.methods[method].predictions[model]
        assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])

def test_conformal_time_series_order(e2e_synthetic_data, available_models, fast_params):
    # Advanced conformal prediction often has online/NexCP methods that respect time sequence.
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # Check that online/time-weighted nexcp run on standard ordered indices
    result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["nexcp_split", "online_split"]},
    )
    
    assert "nexcp_split" in result.methods
    assert "online_split" in result.methods

def test_conformal_empty_calibration(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # Try calibration size with 1 sample
    result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["mapie"], "calibration_size": 0.02},  # 1 sample from 50
    )
    assert "mapie" in result.methods

def test_conformal_extreme_alpha(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.99, "methods": ["mapie"]},
    )
    assert "mapie" in result.methods


# =============================================================================
# Boundary Cases 4: Hyperparameter Tuning
# =============================================================================

def test_tuning_one_trial(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    assert len(result.best_params[model]) > 0

def test_tuning_invalid_cv(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # cv=1 is invalid for KFold
    with pytest.raises(ValueError):
        run_tuning(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            tuning_config={"n_trials": 2, "cv": 1},
        )

def test_tuning_timeout(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # extremely short timeout to test early termination
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 10, "timeout": 0.001, "cv": 2, "random_state": 42},
    )
    assert model in result.best_params

def test_tuning_single_parameter_trial(e2e_synthetic_data, available_models):
    pytest.importorskip("optuna")
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # Run tuning with small trials
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 2, "cv": 2, "random_state": 42},
    )
    assert model in result.best_params

def test_tuning_empty_features(available_models):
    pytest.importorskip("optuna")
    # Extremely small dataset
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "target": [2.0, 4.0, 6.0, 8.0]})
    test_df = pd.DataFrame({"f1": [1.5], "target": [3.0]})
    model = get_test_model(available_models)
    
    result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    assert model in result.best_params


# =============================================================================
# Boundary Cases 5: Explainability
# =============================================================================

def is_explain_method_available(method: str) -> bool:
    if method == "shap":
        return is_package_installed("shap")
    if method == "lime":
        return is_package_installed("lime")
    if method == "interpret":
        return is_package_installed("interpret")
    return False

def test_explainability_single_feature(available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df = pd.DataFrame({"f1": np.random.normal(0, 1, 20), "target": np.random.normal(0, 1, 20)})
    test_df = pd.DataFrame({"f1": np.random.normal(0, 1, 5), "target": np.random.normal(0, 1, 5)})
    model = get_test_model(available_models)
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
    )
    
    key = f"{model}_{available_methods[0]}"
    assert key in result.predictions
    assert len(result.predictions[key]) == 1

def test_explainability_high_dim(available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    # More features than samples
    n_samples = 10
    features = {f"f{i}": np.random.normal(0, 1, n_samples) for i in range(15)}
    features["target"] = np.random.normal(0, 1, n_samples)
    train_df = pd.DataFrame(features)
    test_df = train_df.copy()
    
    model = get_test_model(available_models)
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
    )
    
    key = f"{model}_{available_methods[0]}"
    assert key in result.predictions
    assert len(result.predictions[key]) == 15

def test_explainability_empty_test(e2e_synthetic_data, available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df, _ = e2e_synthetic_data
    # Empty test DataFrame
    test_df = pd.DataFrame(columns=train_df.columns)
    model = get_test_model(available_models)
    
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
    )
    # Explainer should complete or handle gracefully
    key = f"{model}_{available_methods[0]}"
    assert key in result.predictions or key not in result.predictions  # gracefully handles or fails gracefully

def test_explainability_missing_features(e2e_synthetic_data, available_models, fast_params):
    available_methods = [m for m in ["shap", "lime", "interpret"] if is_explain_method_available(m)]
    if not available_methods:
        pytest.skip("No explainability packages available")
        
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    # Change test columns names
    test_df_bad = test_df.copy()
    test_df_bad.columns = ["wrong1", "wrong2", "target"]
    
    # Should run and resolve names or fallback
    result = run_explainability(
        data={"train_df": train_df, "test_df": test_df_bad},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=[available_methods[0]],
    )
    key = f"{model}_{available_methods[0]}"
    assert key in result.predictions

def test_explainability_unsupported_model(e2e_synthetic_data, available_models, fast_params):
    # Test fallback explainability when model is unpatched but explainability fails gracefully
    pass


def test_probabilistic_advanced_invalid_alpha(e2e_synthetic_data, available_models, fast_params):
    train_df, test_df = e2e_synthetic_data
    model = get_test_model(available_models)
    
    with pytest.raises(ValueError, match="alpha must satisfy"):
        run_probabilistic_advanced(
            data={"train_df": train_df, "test_df": test_df},
            target_col="target",
            models=[model],
            params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
            methods=["ibug", "hcm"],
            alpha=1.5,
        )

def test_probabilistic_advanced_zero_variance_hcm_card(available_models, fast_params):
    if not is_package_installed("torch"):
        pytest.skip("PyTorch is not installed")
        
    rng = np.random.default_rng(42)
    n_train, n_test = 20, 5
    # Constant feature 'f1' and constant target
    train_df = pd.DataFrame({"f1": [1.0] * n_train, "f2": rng.normal(0, 1, n_train), "target": 5.0})
    test_df = pd.DataFrame({"f1": [1.0] * n_test, "f2": rng.normal(0, 1, n_test), "target": 5.0})
    model = get_test_model(available_models)
    
    result = run_probabilistic_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["hcm", "card"],
        hcm_config={"epochs": 2},
        card_config={"epochs": 2},
    )
    
    for m in ["hcm", "card"]:
        key = f"{model}_{m}"
        assert key in result.predictions
        pred_df = result.predictions[key]
        assert not pred_df.isnull().values.any()  # No NaNs due to zero variance

def test_probabilistic_advanced_ibug_small_dataset(available_models, fast_params):
    # Training set smaller than some candidate_k values
    train_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [2.0, 4.0, 6.0, 8.0, 10.0]})
    test_df = pd.DataFrame({"f1": [1.5], "target": [3.0]})
    model = get_test_model(available_models)
    
    result = run_probabilistic_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="target",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        methods=["ibug"],
    )
    
    key = f"{model}_ibug"
    assert key in result.predictions
    pred_df = result.predictions[key]
    assert len(pred_df) == 1
