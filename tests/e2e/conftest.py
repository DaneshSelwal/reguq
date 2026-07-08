from __future__ import annotations

import importlib
import pytest
import pandas as pd
import numpy as np

def is_package_installed(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def get_available_models() -> list[str]:
    models = ["randomforest", "gradientboosting"]  # Standard sklearn models always available
    if is_package_installed("lightgbm"):
        models.append("lightgbm")
    if is_package_installed("xgboost"):
        models.append("xgboost")
    if is_package_installed("catboost"):
        models.append("catboost")
    if is_package_installed("ngboost"):
        models.append("ngboost")
    if is_package_installed("pgbm"):
        models.append("pgbm")
    if is_package_installed("gpboost"):
        models.append("gpboost")
    if is_package_installed("pytorch_tabnet") and is_package_installed("torch"):
        models.append("tabnet")
    return models

@pytest.fixture(scope="session")
def available_models() -> list[str]:
    return get_available_models()

@pytest.fixture
def fast_params() -> dict[str, dict[str, any]]:
    """Tiny parameters for real models to make E2E tests run fast."""
    return {
        "lightgbm": {"n_estimators": 2, "num_leaves": 4, "verbose": -1},
        "xgboost": {"n_estimators": 2, "max_depth": 2, "verbosity": 0},
        "catboost": {"iterations": 2, "depth": 2, "verbose": False},
        "ngboost": {"n_estimators": 2, "verbose": False},
        "pgbm": {"max_iter": 2, "verbose": False},
        "randomforest": {"n_estimators": 2, "max_depth": 2},
        "gradientboosting": {"n_estimators": 2, "max_depth": 2},
        "gpboost": {"n_estimators": 2, "verbose": 0},
        "tabnet": {"n_d": 8, "n_a": 8, "n_steps": 3, "verbose": 0},
    }

@pytest.fixture
def e2e_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic dataset for E2E testing."""
    rng = np.random.default_rng(42)
    n_train = 50
    n_test = 15

    def _make(n: int) -> pd.DataFrame:
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 1.5 * x1 - 2.0 * x2 + rng.normal(0, 0.1, n)
        return pd.DataFrame({"feature1": x1, "feature2": x2, "target": y})

    return _make(n_train), _make(n_test)

@pytest.fixture
def e2e_synthetic_paths(tmp_path, e2e_synthetic_data):
    train_df, test_df = e2e_synthetic_data
    train_path = tmp_path / "e2e_train.csv"
    test_path = tmp_path / "e2e_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path
