"""Hyper-parameter tuning with Optuna on **training data only**.

The v1.2 notebooks scored every trial on the test set and picked the best
of ~400 test evaluations, which biases every reported test metric. Here the
objective is K-fold (or forward-chaining time-series) cross-validated RMSE
on the training set; the test set is never seen. Pruners work because the
running CV score is reported after every fold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._base import ArrayLike, as_1d_float
from .._utils import fit_clone, predict_1d, take_rows
from ..models import make_model


def search_space(trial: Any, name: str) -> dict[str, Any]:
    """Default search space for registered model ``name``."""
    n = name.lower()
    if n == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if n == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 20, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if n == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 100, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 2, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 30, log=True),
        }
    if n == "gpboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100, log=True),
        }
    if n in ("gradientboosting",):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        }
    if n == "histgradientboosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 50, 1000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 128, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
        }
    if n in ("randomforest", "extratrees"):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30, log=True),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
    if n == "linear":
        return {"alpha": trial.suggest_float("alpha", 1e-6, 1e3, log=True)}
    raise KeyError(f"no default search space for {name!r}; pass `space=`")


def cv_folds(
    n: int, n_splits: int, time_ordered: bool, random_state: int | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, val_idx) pairs: KFold, or forward-chaining splits if ``time_ordered``."""
    if time_ordered:
        from sklearn.model_selection import TimeSeriesSplit

        return list(TimeSeriesSplit(n_splits=n_splits).split(np.arange(n)))
    from sklearn.model_selection import KFold

    return list(KFold(n_splits, shuffle=True, random_state=random_state).split(np.arange(n)))


@dataclass
class TuningResult:
    model: str
    best_params: dict[str, Any]
    best_cv_rmse: float
    n_trials: int
    n_pruned: int
    trials: list[dict[str, Any]] = field(default_factory=list)


def tune(
    name: str,
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_trials: int = 50,
    n_splits: int = 5,
    time_ordered: bool = False,
    sampler: str = "tpe",
    pruner: str = "median",
    space: Any = None,
    fixed_params: dict[str, Any] | None = None,
    random_state: int | None = 0,
    timeout: float | None = None,
) -> TuningResult:
    """Tune registered model ``name`` by cross-validated RMSE on ``(X, y)`` only.

    ``space`` is an optional callable ``trial -> params``. ``sampler`` is
    ``"tpe"``, ``"random"`` or ``"auto"`` (OptunaHub AutoSampler, needs
    ``optunahub``); ``pruner`` is ``"median"``, ``"hyperband"`` or ``"none"``.
    """
    from ..probabilistic._base import require

    optuna = require("optuna", "tuning")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y_arr = as_1d_float(y)
    folds = cv_folds(len(y_arr), n_splits, time_ordered, random_state)
    space_fn = space or (lambda t: search_space(t, name))

    def objective(trial: Any) -> float:
        params = {**space_fn(trial), **(fixed_params or {})}
        est = make_model(name, params, random_state=random_state)
        sq = []
        for k, (tr, va) in enumerate(folds):
            m = fit_clone(est, take_rows(X, tr), y_arr[tr])
            sq.append(np.mean((y_arr[va] - predict_1d(m, take_rows(X, va))) ** 2))
            running = float(np.sqrt(np.mean(sq)))
            trial.report(running, step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.sqrt(np.mean(sq)))

    if sampler == "tpe":
        smp = optuna.samplers.TPESampler(seed=random_state)
    elif sampler == "random":
        smp = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler == "auto":
        hub = require("optunahub", "tuning")
        smp = hub.load_module("samplers/auto_sampler").AutoSampler(seed=random_state)
    else:
        raise ValueError("sampler must be 'tpe', 'random' or 'auto'")
    prn = {
        "median": optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        "hyperband": optuna.pruners.HyperbandPruner(),
        "none": optuna.pruners.NopPruner(),
    }[pruner]
    study = optuna.create_study(direction="minimize", sampler=smp, pruner=prn)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    return TuningResult(
        model=name,
        best_params={**study.best_params, **(fixed_params or {})},
        best_cv_rmse=float(study.best_value),
        n_trials=len(study.trials),
        n_pruned=len(pruned),
        trials=[
            {"number": t.number, "value": t.value, "state": t.state.name, **t.params}
            for t in study.trials
        ],
    )


__all__ = ["TuningResult", "cv_folds", "search_space", "tune"]
