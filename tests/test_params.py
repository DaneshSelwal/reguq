from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reguq.params import resolve_model_params


def test_resolve_model_params_load_only_from_file(tmp_path):
    params_path = tmp_path / "params.json"
    payload = {
        "lightgbm": {"n_estimators": 55},
        "xgboost": {"n_estimators": 33},
    }
    params_path.write_text(json.dumps(payload), encoding="utf-8")

    X_train = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10)})
    y_train = pd.Series(np.arange(10))

    resolved, tuned = resolve_model_params(
        models=["lightgbm", "xgboost"],
        params_source={"mode": "load_only", "params_path": str(params_path)},
        X_train=X_train,
        y_train=y_train,
    )

    assert tuned == {}
    assert resolved["lightgbm"]["n_estimators"] == 55
    assert resolved["xgboost"]["n_estimators"] == 33


def test_resolve_model_params_load_or_tune_calls_tuner(monkeypatch):
    X_train = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10)})
    y_train = pd.Series(np.arange(10))

    calls = []

    def _fake_tune(model_id, X_train, y_train, tuning_config=None):
        calls.append(model_id)
        return {"depth": 3}, 0.5

    monkeypatch.setattr("reguq.params.tune_single_model", _fake_tune)

    resolved, tuned = resolve_model_params(
        models=["lightgbm", "xgboost"],
        params_source={"mode": "load_or_tune", "params": {"lightgbm": {"n_estimators": 10}}},
        X_train=X_train,
        y_train=y_train,
    )

    assert resolved["lightgbm"]["n_estimators"] == 10
    assert "xgboost" in calls
    assert tuned["xgboost"]["depth"] == 3


def test_resolve_model_params_load_only_missing_raises():
    X_train = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10)})
    y_train = pd.Series(np.arange(10))

    with pytest.raises(ValueError):
        resolve_model_params(
            models=["lightgbm"],
            params_source={"mode": "load_only", "params": {}},
            X_train=X_train,
            y_train=y_train,
        )
