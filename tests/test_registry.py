from __future__ import annotations

import pytest

import reguq.registry as registry


def test_validate_models_unknown_model_raises():
    with pytest.raises(ValueError):
        registry.validate_models(["does_not_exist"], phase="tuning")


def test_validate_models_phase_support():
    with pytest.raises(ValueError, match="does not support phase 'quantile'"):
        registry.validate_models(["hcm"], phase="quantile")


def test_list_supported_models_for_quantile_contains_expected_models():
    supported = registry.list_supported_models("quantile")
    assert "lightgbm" in supported
    assert "xgboost" in supported
    assert "catboost" in supported
    assert "ngboost" in supported
    assert "pgbm" in supported
    assert "tabnet" in supported
    assert "randomforest" in supported
    assert "hcm" not in supported
