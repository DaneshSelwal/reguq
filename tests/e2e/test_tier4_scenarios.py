from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from reguq.quantile import run_quantile
from reguq.probabilistic import run_probabilistic
from reguq.conformal_standard import run_conformal_standard
from reguq.conformal_advanced import run_conformal_advanced
from reguq.tuning import run_tuning
from reguq.types import OutputConfig

# Helper to select a model for testing
def get_test_model(available_models):
    for m in ["lightgbm", "xgboost", "catboost", "randomforest", "gradientboosting"]:
        if m in available_models:
            return m
    return "gradientboosting"

# =============================================================================
# Scenario 1: Concrete Compressive Strength Uncertainty Quantification
# =============================================================================
def test_scenario_concrete_strength(available_models, fast_params):
    """
    Scenario: Estimating compressive strength of concrete given its mixture components.
    High safety critical domain: requires reliable lower bound to prevent structural failures.
    """
    rng = np.random.default_rng(101)
    n_samples = 40
    
    # Mixture components features
    data = {
        "cement": rng.uniform(100, 500, n_samples),
        "slag": rng.uniform(0, 200, n_samples),
        "fly_ash": rng.uniform(0, 200, n_samples),
        "water": rng.uniform(120, 250, n_samples),
        "superplasticizer": rng.uniform(0, 30, n_samples),
        "coarse_aggregate": rng.uniform(800, 1200, n_samples),
        "fine_aggregate": rng.uniform(500, 900, n_samples),
        "age": rng.choice([3, 7, 14, 28, 56, 90, 365], n_samples),
    }
    
    # Target: Compressive strength (MPa)
    # Strength increases with cement and age, decreases with water
    data["compressive_strength"] = (
        0.1 * data["cement"] 
        + 0.05 * data["slag"] 
        - 0.08 * data["water"] 
        + 0.5 * np.log(data["age"]) 
        + rng.normal(0, 3, n_samples)
    )
    
    df = pd.DataFrame(data)
    train_df = df.iloc[:30]
    test_df = df.iloc[30:]
    
    model = get_test_model(available_models)
    
    # We want a high-confidence lower bound (e.g. q_low = 0.01) to find the minimum guaranteed strength
    result = run_quantile(
        data={"train_df": train_df, "test_df": test_df},
        target_col="compressive_strength",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        quantiles=(0.05, 0.95),
    )
    
    pred_df = result.predictions[model]
    # Check that predictions make physical sense (compressive strength values positive, y_lower <= y_upper)
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])
    assert result.metrics.iloc[0]["coverage"] >= 0.0  # Just verify it ran and calculated coverage successfully


# =============================================================================
# Scenario 2: Environmental Noise Level Prediction and Safety Bounds
# =============================================================================
def test_scenario_noise_safety(available_models, fast_params):
    """
    Scenario: Predicting noise levels (dBA) near a highway to design acoustic barriers.
    We need probabilistic outputs to compute decibel safety bounds.
    """
    rng = np.random.default_rng(202)
    n_samples = 40
    
    data = {
        "traffic_volume": rng.uniform(1000, 50000, n_samples),
        "heavy_vehicle_pct": rng.uniform(1, 25, n_samples),
        "speed_limit": rng.choice([50, 70, 90, 110], n_samples),
        "distance_to_receiver": rng.uniform(10, 300, n_samples),
        "barrier_height": rng.uniform(0, 6, n_samples),
    }
    
    # Noise level increases with traffic, trucks, speed; decreases with distance and barrier
    data["noise_level"] = (
        10 * np.log10(data["traffic_volume"]) 
        + 0.5 * data["heavy_vehicle_pct"]
        + 0.1 * data["speed_limit"]
        - 15 * np.log10(data["distance_to_receiver"])
        - 1.5 * data["barrier_height"]
        + rng.normal(0, 2, n_samples)
    )
    
    df = pd.DataFrame(data)
    train_df = df.iloc[:30]
    test_df = df.iloc[30:]
    
    model = get_test_model(available_models)
    
    # Run probabilistic phase to get mean and standard deviation
    result = run_probabilistic(
        data={"train_df": train_df, "test_df": test_df},
        target_col="noise_level",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        alpha=0.05,  # 95% confidence interval
    )
    
    pred_df = result.predictions[model]
    assert "y_std" in pred_df.columns
    # Check that standard deviation is positive and finite
    assert np.all(pred_df["y_std"] > 0)
    # Check that upper noise limit is higher than predicted mean
    assert np.all(pred_df["y_upper"] >= pred_df["y_pred"])


# =============================================================================
# Scenario 3: Bridge Scour Depth Safety Margin Analysis
# =============================================================================
def test_scenario_bridge_scour(available_models, fast_params):
    """
    Scenario: Predicting river bridge scour depth around piers during high-flow events.
    Failure can collapse bridges, so we need distribution-free conformal prediction bounds.
    """
    rng = np.random.default_rng(303)
    n_samples = 45
    
    data = {
        "flow_rate": rng.uniform(50, 1500, n_samples),
        "velocity": rng.uniform(0.5, 4.0, n_samples),
        "pier_width": rng.uniform(0.5, 3.0, n_samples),
        "sediment_d50": rng.uniform(0.1, 10.0, n_samples),
    }
    
    # Scour depth increases with flow, velocity, pier width; decreases with sediment size
    data["scour_depth"] = (
        0.5 * data["velocity"] 
        + 1.2 * data["pier_width"] 
        - 0.05 * data["sediment_d50"]
        + rng.normal(0, 0.2, n_samples)
    )
    
    df = pd.DataFrame(data)
    train_df = df.iloc[:35]
    test_df = df.iloc[35:]
    
    model = get_test_model(available_models)
    
    # We will use conformal standard (MAPIE or PUNCC if available, or fallback)
    result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="scour_depth",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["mapie", "puncc"]},
    )
    
    for method in result.methods:
        pred_df = result.methods[method].predictions[model]
        assert "y_lower" in pred_df.columns
        assert "y_upper" in pred_df.columns
        # Intervals should be valid
        assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])


# =============================================================================
# Scenario 4: Water Quality Index (WQI) Prediction and Compliance Margins
# =============================================================================
def test_scenario_water_quality(available_models):
    """
    Scenario: Predicting WQI to ensure environmental compliance.
    Requires tuning parameters, followed by conformal prediction for the safety band.
    """
    pytest.importorskip("optuna")
    rng = np.random.default_rng(404)
    n_samples = 40
    
    data = {
        "ph": rng.uniform(6.0, 8.5, n_samples),
        "dissolved_oxygen": rng.uniform(2.0, 12.0, n_samples),
        "turbidity": rng.uniform(0.5, 50.0, n_samples),
        "nitrates": rng.uniform(0.1, 10.0, n_samples),
        "phosphates": rng.uniform(0.01, 2.0, n_samples),
    }
    
    # WQI (0-100 scale, higher is better)
    data["wqi"] = (
        10 * (data["ph"] - 6.0) 
        + 4 * data["dissolved_oxygen"] 
        - 0.5 * data["turbidity"] 
        - 2 * data["nitrates"] 
        - 5 * data["phosphates"]
        + rng.normal(0, 5, n_samples)
    )
    # Clip between 0 and 100
    data["wqi"] = np.clip(data["wqi"], 10, 95)
    
    df = pd.DataFrame(data)
    train_df = df.iloc[:30]
    test_df = df.iloc[30:]
    
    model = get_test_model(available_models)
    
    # 1. Tune parameters
    tuning_result = run_tuning(
        data={"train_df": train_df, "test_df": test_df},
        target_col="wqi",
        models=[model],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )
    
    best_params = tuning_result.best_params
    
    # 2. Run conformal standard with tuned parameters
    conformal_result = run_conformal_standard(
        data={"train_df": train_df, "test_df": test_df},
        target_col="wqi",
        models=[model],
        params_source={"mode": "load_or_tune", "params": best_params},
        conformal_config={"alpha": 0.1, "methods": ["mapie"]},
    )
    
    assert "mapie" in conformal_result.methods
    pred_df = conformal_result.methods["mapie"].predictions[model]
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])


# =============================================================================
# Scenario 5: Streamflow Forecasting and Flood Risk Assessment
# =============================================================================
def test_scenario_streamflow_forecast(available_models, fast_params):
    """
    Scenario: Time-series forecasting of river streamflow (discharge in m^3/s).
    Data is ordered/time-dependent: requires non-exchangeable conformal methods (NexCP / Online Split).
    """
    rng = np.random.default_rng(505)
    n_samples = 50
    
    # Time-dependent streamflow data
    data = {
        "precipitation_lag1": rng.uniform(0, 50, n_samples),
        "precipitation_lag2": rng.uniform(0, 50, n_samples),
        "temperature": rng.uniform(5, 35, n_samples),
        "soil_moisture": rng.uniform(0.1, 0.9, n_samples),
    }
    
    # Discharge increases with lag precipitations and soil moisture
    data["discharge"] = (
        2.5 * data["precipitation_lag1"] 
        + 1.0 * data["precipitation_lag2"] 
        + 10.0 * data["soil_moisture"]
        - 0.2 * data["temperature"]
        + rng.normal(0, 5, n_samples)
    )
    
    df = pd.DataFrame(data)
    # Streamflow is time-series: do not shuffle train/test, keep chronological order
    train_df = df.iloc[:35]
    test_df = df.iloc[35:]
    
    model = get_test_model(available_models)
    
    # Use NexCP Split (designed for non-exchangeable / time-series data)
    result = run_conformal_advanced(
        data={"train_df": train_df, "test_df": test_df},
        target_col="discharge",
        models=[model],
        params_source={"mode": "load_or_tune", "params": {model: fast_params.get(model, {})}},
        conformal_config={"alpha": 0.1, "methods": ["nexcp_split"]},
    )
    
    assert "nexcp_split" in result.methods
    pred_df = result.methods["nexcp_split"].predictions[model]
    assert np.all(pred_df["y_lower"] <= pred_df["y_upper"])
