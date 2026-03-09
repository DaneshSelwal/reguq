from __future__ import annotations

import pytest

import reguq.registry as registry


def test_validate_models_unknown_model_raises():
    with pytest.raises(ValueError):
        registry.validate_models(["does_not_exist"], phase="tuning")


def test_validate_models_phase_support():
    with pytest.raises(ValueError):
        registry.validate_models(["ngboost"], phase="quantile")


def test_list_supported_models_for_quantile_contains_expected_models():
    supported = registry.list_supported_models("quantile")
    assert "lightgbm" in supported
    assert "xgboost" in supported
    assert "catboost" in supported
    assert "ngboost" not in supported
