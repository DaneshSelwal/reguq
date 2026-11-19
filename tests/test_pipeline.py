import numpy as np
import pandas as pd
import pytest

from conftest import make_linear
from reguq.models import available_models, make_model
from reguq.pipeline import PipelineConfig, load_data, run_pipeline


@pytest.fixture
def csvs(tmp_path):
    rng = np.random.default_rng(0)
    X, y = make_linear(500, rng, hetero=True)
    cols = [" a ", "b", "c"]
    tr = pd.DataFrame(X[:400], columns=cols).assign(**{"target ": y[:400]})
    te = pd.DataFrame(X[400:], columns=cols).assign(target=y[400:])
    tr.to_csv(tmp_path / "train.csv", index=False)
    te.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


def test_load_strips_whitespace_and_detects_target(csvs):
    cfg = PipelineConfig(str(csvs / "train.csv"), str(csvs / "test.csv"))
    Xtr, ytr, Xte, yte = load_data(cfg)
    assert list(Xtr.columns) == ["a", "b", "c"] and ytr.size == 400 and yte.size == 100


def test_mismatched_target_names_raise(tmp_path, csvs):
    te = pd.read_csv(csvs / "test.csv").rename(columns={"target": "other"})
    te.to_csv(tmp_path / "test2.csv", index=False)
    with pytest.raises(KeyError, match="test_target"):
        load_data(PipelineConfig(str(csvs / "train.csv"), str(tmp_path / "test2.csv")))


def test_registry_core_models():
    assert {"linear", "randomforest", "gradientboosting"} <= set(available_models())
    assert make_model("randomforest", {"n_estimators": 7}, random_state=3).random_state == 3


def test_pipeline_runs_and_reports_errors(csvs):
    out = csvs / "out"
    cfg = PipelineConfig(
        str(csvs / "train.csv"),
        str(csvs / "test.csv"),
        models=["linear", "gradientboosting"],
        methods=["split", "cqr", "cv_plus", "aci", "acp", "ibug"],
        tuning=None,
        output_dir=str(out),
    )
    res = run_pipeline(cfg, log=lambda s: None)
    m = res.metrics.set_index(["method", "model"])
    # Ridge has no quantile objective and no leaves: must be reported, not faked.
    assert m.loc[("cqr", "linear"), "status"].startswith("error: TypeError")
    assert m.loc[("ibug", "linear"), "status"].startswith("error: TypeError")
    ok = res.metrics[res.metrics.status == "ok"]
    assert len(ok) >= 9
    assert (ok.coverage > 0.75).all()
    assert (out / "metrics.csv").exists() and (out / "predictions").is_dir()


def test_tuning_uses_train_only_and_prunes(csvs):
    pytest.importorskip("optuna")
    from reguq.tuning import tune

    Xtr, ytr, _, _ = load_data(PipelineConfig(str(csvs / "train.csv"), str(csvs / "test.csv")))
    res = tune(
        "gradientboosting",
        Xtr,
        ytr,
        n_trials=12,
        n_splits=3,
        random_state=0,
        space=lambda t: {
            "n_estimators": t.suggest_int("n_estimators", 5, 200, log=True),
            "max_depth": t.suggest_int("max_depth", 1, 4),
        },
    )
    assert res.n_trials == 12 and res.best_cv_rmse > 0
