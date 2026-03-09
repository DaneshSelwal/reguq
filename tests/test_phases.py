from __future__ import annotations

import pytest

from reguq.conformal_standard import run_conformal_standard
from reguq.probabilistic import run_probabilistic
from reguq.quantile import run_quantile
from reguq.tuning import run_tuning


def test_run_quantile_smoke(synthetic_paths, patched_estimators):
    train_path, test_path = synthetic_paths
    result = run_quantile(
        data={"train_path": train_path, "test_path": test_path},
        target_col="target",
        models=["lightgbm", "xgboost", "catboost"],
        params_source={"mode": "defaults"},
    )

    assert not result.metrics.empty
    assert set(result.predictions.keys()) == {"lightgbm", "xgboost", "catboost"}
    assert {"coverage", "avg_interval_width"}.issubset(set(result.metrics.columns))


def test_run_probabilistic_smoke(synthetic_paths, patched_estimators):
    train_path, test_path = synthetic_paths
    result = run_probabilistic(
        data={"train_path": train_path, "test_path": test_path},
        target_col="target",
        models=["lightgbm", "xgboost", "catboost", "ngboost", "pgbm"],
        params_source={"mode": "defaults"},
    )

    assert not result.metrics.empty
    assert set(result.predictions.keys()) == {"lightgbm", "xgboost", "catboost", "ngboost", "pgbm"}
    assert {"nll", "crps"}.issubset(set(result.metrics.columns))


def test_run_conformal_standard_smoke(synthetic_paths, patched_estimators):
    train_path, test_path = synthetic_paths
    result = run_conformal_standard(
        data={"train_path": train_path, "test_path": test_path},
        target_col="target",
        models=["lightgbm", "xgboost", "catboost", "ngboost", "pgbm"],
        params_source={"mode": "defaults"},
        conformal_config={"alpha": 0.1, "methods": ["mapie", "puncc"]},
    )

    assert "mapie" in result.methods
    assert "puncc" in result.methods
    assert not result.methods["mapie"].metrics.empty
    assert not result.methods["puncc"].metrics.empty


def test_run_tuning_smoke(synthetic_paths, patched_estimators):
    pytest.importorskip("optuna")
    train_path, test_path = synthetic_paths
    result = run_tuning(
        data={"train_path": train_path, "test_path": test_path},
        target_col="target",
        models=["lightgbm", "xgboost"],
        tuning_config={"n_trials": 1, "cv": 2, "random_state": 42},
    )

    assert set(result.best_params.keys()) == {"lightgbm", "xgboost"}
    assert not result.summary.empty


def test_inline_plot_path_does_not_break(synthetic_paths, patched_estimators, monkeypatch, tmp_path):
    import reguq.charts as charts

    calls: list[bool] = []

    def _fake_display(fig, enabled: bool):
        calls.append(enabled)

    monkeypatch.setattr(charts, "_display_inline", _fake_display)

    train_path, test_path = synthetic_paths
    result = run_quantile(
        data={"train_path": train_path, "test_path": test_path},
        target_col="target",
        models=["lightgbm"],
        params_source={"mode": "defaults"},
        output_config={
            "output_dir": str(tmp_path / "inline"),
            "export_excel": False,
            "export_plots": False,
            "show_inline_plots": True,
        },
    )

    assert not result.metrics.empty
    assert any(calls)
    assert not any(p.suffix == ".png" for p in result.artifacts)
