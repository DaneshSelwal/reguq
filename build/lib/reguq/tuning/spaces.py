"""Default Optuna search spaces per registered model."""

from __future__ import annotations

from typing import Any


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
