"""Pipeline runner."""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import pandas as pd

from ..models import make_model
from .config import PipelineConfig
from .data import load_data
from .evaluate import evaluate
from .export import export
from .methods import METHODS, MODEL_FREE

logger = logging.getLogger("reguq.pipeline")


def _build_wrapped(
    method: str, model: str, params: dict[str, Any], cfg: PipelineConfig, mp: dict[str, Any]
) -> Any:
    return METHODS[method](make_model(model, params, random_state=cfg.random_state), cfg, mp)


@dataclass
class PipelineResult:
    """Metrics table (one row per model x method), predictions, parameters, tuning log."""

    metrics: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    best_params: dict[str, dict[str, Any]]
    tuning: dict[str, Any]


def run_pipeline(
    cfg: PipelineConfig, *, log: Callable[[str], None] | None = None
) -> PipelineResult:
    """Run the configured models x methods and return (and optionally export) the results.

    Steps: load data -> tune each model on the training data only -> fit
    every method on the training data (each method holds out its own
    calibration data) -> evaluate once on the test set.
    """
    from ..tuning import tune

    log = log or logger.info

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
                partial(_build_wrapped, method, name, best[name], cfg, mp),
            )
    for method in cfg.model_free_methods:
        if method not in MODEL_FREE:
            raise KeyError(f"unknown model-free method {method!r}; available: {sorted(MODEL_FREE)}")
        mp = cfg.method_params.get(method, {})
        run_one(method, "-", partial(MODEL_FREE[method], cfg, mp))

    result = PipelineResult(pd.DataFrame(rows), preds, best, tuning_info)
    if cfg.output_dir:
        export(result, cfg)
        log(f"exported to {cfg.output_dir}")
    return result
