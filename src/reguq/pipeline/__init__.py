"""End-to-end pipeline: load -> tune on train only -> fit UQ methods -> evaluate on test.

Design rules (each fixes a v1.2 problem):

* the test set is used **once**, for final evaluation - never for tuning,
  early stopping or calibration;
* hyper-parameters are tuned once per model and shared by every method;
* a method that fails is reported with its error message - nothing silently
  falls back to a different method under the same label;
* every method is evaluated with the same metric functions.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import conformal as C
from .. import probabilistic as P
from .. import quantile as Q
from ..metrics import calibration_error, interval_report, point_report
from ..models import make_model


# ------------------------------------------------------------------ config
@dataclass
class PipelineConfig:
    """Configuration of one pipeline run (all paths may be strings)."""

    train_path: str
    test_path: str
    target: str | None = None  # default: last column
    test_target: str | None = None  # only if the test file names the target differently
    features: list[str] | None = None
    time_ordered: bool = False
    alpha: float = 0.1
    calib_size: float = 0.25
    random_state: int = 0
    models: list[str] = field(
        default_factory=lambda: [
            "lightgbm",
            "xgboost",
            "catboost",
            "gradientboosting",
            "randomforest",
        ]
    )
    methods: list[str] = field(
        default_factory=lambda: [
            "split",
            "normalized",
            "cqr",
            "cv_plus",
            "jab",
            "multi_quantile",
            "ibug",
        ]
    )
    model_free_methods: list[str] = field(default_factory=list)  # e.g. ["ngboost", "qrf", "hcm"]
    tuning: dict[str, Any] | None = field(default_factory=lambda: {"n_trials": 30, "n_splits": 5})
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    method_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_dir: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            from ..probabilistic._base import require

            data = require("yaml", "report").safe_load(text)
        else:
            data = json.loads(text)
        return cls(**data)


# -------------------------------------------------------------------- data
def load_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Read train/test CSVs, strip column-name whitespace, align features."""
    tr = pd.read_csv(cfg.train_path)
    te = pd.read_csv(cfg.test_path)
    tr.columns = [str(c).strip() for c in tr.columns]
    te.columns = [str(c).strip() for c in te.columns]
    target = (cfg.target or tr.columns[-1]).strip()
    test_target = (cfg.test_target or target).strip()
    if target not in tr.columns:
        raise KeyError(f"target {target!r} not in training columns {list(tr.columns)}")
    if test_target not in te.columns:
        raise KeyError(
            f"target {test_target!r} not in test columns {list(te.columns)}. If the test file "
            "uses a different target name, set `test_target` explicitly (and note that the "
            "test then measures a different quantity)."
        )
    feats = cfg.features or [c for c in tr.columns if c != target]
    missing = [c for c in feats if c not in te.columns]
    if missing:
        raise KeyError(f"test file lacks features {missing}")
    return tr[feats], tr[target].to_numpy(float), te[feats], te[test_target].to_numpy(float)


# ------------------------------------------------------------ method table
Builder = Callable[[Any, PipelineConfig, dict[str, Any]], Any]


def _cal(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "calib_size": cfg.calib_size,
        "time_ordered": cfg.time_ordered,
        "random_state": cfg.random_state,
    }


METHODS: dict[str, Builder] = {
    "split": lambda m, c, p: C.SplitConformalRegressor(m, **_cal(c), **p),
    "normalized": lambda m, c, p: C.SplitConformalRegressor(m, score="normalized", **_cal(c), **p),
    "cqr": lambda m, c, p: C.ConformalizedQuantileRegressor(m, alpha=c.alpha, **_cal(c), **p),
    "cv_plus": lambda m, c, p: C.CrossConformalRegressor(
        m, shuffle=not c.time_ordered, random_state=c.random_state, **p
    ),
    "jab": lambda m, c, p: C.JackknifePlusAfterBootstrapRegressor(
        m, random_state=c.random_state, **p
    ),
    "nexcp": lambda m, c, p: C.NexCPRegressor(m, calib_size=c.calib_size, **p),
    "aci": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="aci", alpha=c.alpha, method_params=p or None
    ),
    "dtaci": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="dtaci", alpha=c.alpha, method_params=p or None
    ),
    "sf_ogd": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="sf_ogd", alpha=c.alpha, method_params=p or None
    ),
    "saocp": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="saocp", alpha=c.alpha, method_params=p or None
    ),
    "nexcp_online": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="nexcp", alpha=c.alpha, method_params=p or None
    ),
    "extreme": lambda m, c, p: C.ExtremeValueConformalRegressor(m, **_cal(c), **p),
    "acp": lambda m, c, p: C.AdaptiveCoveragePolicyRegressor(m, alpha=c.alpha, **_cal(c), **p),
    "multi_quantile": lambda m, c, p: Q.MultiQuantileRegressor(m, **p),
    "ibug": lambda m, c, p: P.IBUGRegressor(
        m, time_ordered=c.time_ordered, random_state=c.random_state, **p
    ),
    "card": lambda m, c, p: P.CARDRegressor(m, random_state=c.random_state, **p),
}

MODEL_FREE: dict[str, Callable[[PipelineConfig, dict[str, Any]], Any]] = {
    "ngboost": lambda c, p: P.NGBoostRegressor(random_state=c.random_state, **p),
    "pgbm": lambda c, p: P.PGBMRegressor(random_state=c.random_state, **p),
    "hcm": lambda c, p: P.HCMRegressor(
        time_ordered=c.time_ordered, random_state=c.random_state, **p
    ),
    "treeffuser": lambda c, p: P.TreeffuserRegressor(seed=c.random_state, **p),
    "qrf": lambda c, p: Q.QuantileForestRegressor(random_state=c.random_state, **p),
}


# ------------------------------------------------------------- evaluation
def evaluate(
    est: Any, X_test: Any, y_test: np.ndarray, alpha: float
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Point, interval and (if available) distributional metrics on the test set."""
    online = hasattr(est, "predict_interval_online")
    iv = (
        est.predict_interval_online(X_test, y_test)
        if online
        else est.predict_interval(X_test, alpha)
    )
    pred = est.predict(X_test)
    row: dict[str, Any] = {**point_report(y_test, pred), **interval_report(y_test, iv, alpha)}
    if hasattr(est, "predict_dist"):
        d = est.predict_dist(X_test)
        row["crps"] = d.crps(y_test)
        row["nll"] = d.nll(y_test)
        row.update({f"pit_{k}": v for k, v in calibration_error(d.pit(y_test)).items()})
    if hasattr(est, "alpha_tilde_"):
        row["alpha_tilde"] = est.alpha_tilde_
    preds = pd.DataFrame({"y_true": y_test, "y_pred": pred, "lower": iv[:, 0], "upper": iv[:, 1]})
    return row, preds


@dataclass
class PipelineResult:
    metrics: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    best_params: dict[str, dict[str, Any]]
    tuning: dict[str, Any]


def run_pipeline(cfg: PipelineConfig, *, log: Callable[[str], None] = print) -> PipelineResult:
    """Run the configured models x methods and return / export the results."""
    from ..tuning import tune

    X_tr, y_tr, X_te, y_te = load_data(cfg)
    log(
        f"data: train {X_tr.shape}, test {X_te.shape}, alpha={cfg.alpha}, time_ordered={cfg.time_ordered}"
    )
    best: dict[str, dict[str, Any]] = {}
    tuning_info: dict[str, Any] = {}
    for name in cfg.models:
        if name in cfg.params:
            best[name] = dict(cfg.params[name])
        elif cfg.tuning:
            t0 = time.time()
            res = tune(
                name,
                X_tr,
                y_tr,
                time_ordered=cfg.time_ordered,
                random_state=cfg.random_state,
                **cfg.tuning,
            )
            best[name] = res.best_params
            tuning_info[name] = {
                "best_cv_rmse": res.best_cv_rmse,
                "n_trials": res.n_trials,
                "n_pruned": res.n_pruned,
                "seconds": round(time.time() - t0, 1),
            }
            log(
                f"tuned {name}: cv_rmse={res.best_cv_rmse:.4g} ({res.n_trials} trials, {res.n_pruned} pruned)"
            )
        else:
            best[name] = {}

    rows: list[dict[str, Any]] = []
    preds: dict[str, pd.DataFrame] = {}

    def run_one(method: str, model_name: str, build: Callable[[], Any]) -> None:
        t0 = time.time()
        row: dict[str, Any] = {"method": method, "model": model_name}
        try:
            est = build().fit(X_tr, y_tr)
            metrics, p = evaluate(est, X_te, y_te, cfg.alpha)
            row.update(metrics, status="ok")
            preds[f"{method}__{model_name}"] = p
        except Exception as e:  # recorded, never hidden
            row.update(status=f"error: {type(e).__name__}: {e}")
            row["traceback"] = traceback.format_exc(limit=3)
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        log(
            f"{method:>15} | {model_name:<20} | {row['status'][:60]} | cov={row.get('coverage', float('nan')):.3f}"
        )

    for name in cfg.models:
        for method in cfg.methods:
            if method not in METHODS:
                raise KeyError(f"unknown method {method!r}; available: {sorted(METHODS)}")
            mp = cfg.method_params.get(method, {})
            run_one(
                method,
                name,
                lambda m=method, n=name, mp=mp: METHODS[m](
                    make_model(n, best[n], random_state=cfg.random_state), cfg, mp
                ),
            )
    for method in cfg.model_free_methods:
        if method not in MODEL_FREE:
            raise KeyError(f"unknown model-free method {method!r}; available: {sorted(MODEL_FREE)}")
        mp = cfg.method_params.get(method, {})
        run_one(method, "-", lambda m=method, mp=mp: MODEL_FREE[m](cfg, mp))

    result = PipelineResult(pd.DataFrame(rows), preds, best, tuning_info)
    if cfg.output_dir:
        export(result, cfg)
    return result


def export(result: PipelineResult, cfg: PipelineConfig) -> Path:
    """Write metrics.csv, predictions/*.csv, params.json and config.json."""
    out = Path(cfg.output_dir)  # type: ignore[arg-type]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out / "metrics.csv", index=False)
    for key, df in result.predictions.items():
        df.to_csv(out / "predictions" / f"{key}.csv", index=False)
    (out / "params.json").write_text(
        json.dumps(
            {"best_params": result.best_params, "tuning": result.tuning}, indent=2, default=str
        )
    )
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return out


def main(argv: list[str] | None = None) -> None:
    """CLI: ``reguq-run config.yaml``."""
    import argparse

    ap = argparse.ArgumentParser(prog="reguq-run", description=__doc__)
    ap.add_argument("config", help="YAML or JSON PipelineConfig")
    args = ap.parse_args(argv)
    res = run_pipeline(PipelineConfig.from_file(args.config))
    cols = [
        c
        for c in ["method", "model", "coverage", "mean_width", "interval_score", "rmse", "status"]
        if c in res.metrics
    ]
    print(res.metrics[cols].to_string(index=False))


__all__ = [
    "METHODS",
    "MODEL_FREE",
    "PipelineConfig",
    "PipelineResult",
    "evaluate",
    "export",
    "load_data",
    "main",
    "run_pipeline",
]
