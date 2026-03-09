from __future__ import annotations

from pathlib import Path

import yaml

from reguq.runner import run_from_config


def test_run_from_config_smoke(tmp_path: Path, synthetic_paths, patched_estimators):
    train_path, test_path = synthetic_paths
    config = {
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_col": "target",
        },
        "models": ["lightgbm", "xgboost"],
        "phases": ["quantile", "probabilistic", "conformal_standard"],
        "params_source": {"mode": "defaults"},
        "output": {
            "output_dir": str(tmp_path / "runner_outputs"),
            "export_excel": True,
            "save_json": True,
        },
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_result = run_from_config(config_path)

    assert run_result.output_dir is not None
    assert "quantile" in run_result.results
    assert "probabilistic" in run_result.results
    assert "conformal_standard" in run_result.results

    assert (run_result.output_dir / "quantile.xlsx").exists()
    assert (run_result.output_dir / "probabilistic.xlsx").exists()
    assert (run_result.output_dir / "conformal_standard.xlsx").exists()


def test_run_from_config_with_tuning_merges_params(
    tmp_path: Path,
    synthetic_paths,
    patched_estimators,
    monkeypatch,
):
    train_path, test_path = synthetic_paths

    def _fake_tune(*args, **kwargs):
        return {"n_estimators": 12}, 0.42

    monkeypatch.setattr("reguq.tuning.tune_single_model", _fake_tune)

    config = {
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "target_col": "target",
        },
        "models": ["lightgbm"],
        "phases": ["tuning", "quantile"],
        "params_source": {"mode": "load_or_tune", "params": {}},
        "output": {
            "output_dir": str(tmp_path / "runner_outputs_tuning"),
            "export_excel": False,
            "save_json": True,
        },
    }

    run_result = run_from_config(config)
    tuning_result = run_result.results["tuning"]
    quantile_result = run_result.results["quantile"]

    assert tuning_result.best_params["lightgbm"]["n_estimators"] == 12
    assert quantile_result.params["lightgbm"]["n_estimators"] == 12
    assert (run_result.output_dir / "best_params.json").exists()
