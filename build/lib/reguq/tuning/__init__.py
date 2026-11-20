"""Hyper-parameter tuning with Optuna on **training data only**.

The v1.2 notebooks scored every trial on the test set and picked the best
of ~400 test evaluations, which biases every reported test metric. Here the
objective is K-fold (or forward-chaining time-series) cross-validated RMSE
on the training set; the test set is never seen. Pruners work because the
running CV score is reported after every fold.
"""

from __future__ import annotations

from .optuna import TuningResult, cv_folds, tune
from .spaces import search_space

__all__ = ["TuningResult", "cv_folds", "search_space", "tune"]
