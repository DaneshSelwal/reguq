from __future__ import annotations

from pathlib import Path

from reguq.config import coerce_output_config, coerce_params_source, coerce_split_config, load_config


def test_load_config_from_yaml(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
models:
  - lightgbm
output:
  export_excel: true
""", encoding="utf-8")

    cfg = load_config(config_path)
    assert cfg["models"] == ["lightgbm"]
    assert cfg["output"]["export_excel"] is True


def test_coerce_config_helpers():
    split = coerce_split_config({"test_size": 0.3, "shuffle": True, "random_state": 7})
    assert split.test_size == 0.3
    assert split.shuffle is True
    assert split.random_state == 7

    output = coerce_output_config({"output_dir": "outputs/demo", "export_excel": True})
    assert output.output_dir == "outputs/demo"
    assert output.export_excel is True

    params = coerce_params_source({"mode": "load_only", "params": {"lightgbm": {"n_estimators": 10}}})
    assert params.mode == "load_only"
    assert params.params["lightgbm"]["n_estimators"] == 10
