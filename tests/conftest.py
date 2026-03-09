from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


@pytest.fixture
def synthetic_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n_train = 160
    n_test = 60

    def _make(n: int) -> pd.DataFrame:
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)
        y = 2.1 * x1 - 1.3 * x2 + 0.7 * x3 + rng.normal(0, 0.4, n)
        return pd.DataFrame({"f1": x1, "f2": x2, "f3": x3, "target": y})

    return _make(n_train), _make(n_test)


@pytest.fixture
def synthetic_paths(tmp_path, synthetic_dataframes):
    train_df, test_df = synthetic_dataframes
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


@pytest.fixture
def patched_estimators(monkeypatch):
    import reguq.registry as registry

    def _fake_build_estimator(model_id, phase, params=None, quantile=None):
        if phase == "quantile":
            alpha = 0.5 if quantile is None else float(quantile)
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=60,
                random_state=42,
            )

        return RandomForestRegressor(n_estimators=40, random_state=42)

    monkeypatch.setattr(registry, "build_estimator", _fake_build_estimator)
    return _fake_build_estimator
