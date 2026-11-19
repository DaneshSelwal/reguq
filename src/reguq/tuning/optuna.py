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
from .spaces import search_space


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


__all__ = ["TuningResult", "cv_folds", "tune"]
