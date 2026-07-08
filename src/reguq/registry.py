"""Model registry with phase capability checks and constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import (
    CORE_MODELS,
    PHASE_CONFORMAL_STANDARD,
    PHASE_CONFORMAL_ADVANCED,
    PHASE_EXPLAINABILITY,
    PHASE_PROBABILISTIC,
    PHASE_QUANTILE,
    PHASE_TUNING,
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    phases: frozenset[str]


_MODEL_SPECS: dict[str, ModelSpec] = {
    "lightgbm": ModelSpec(
        model_id="lightgbm",
        display_name="LightGBM",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "xgboost": ModelSpec(
        model_id="xgboost",
        display_name="XGBoost",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "catboost": ModelSpec(
        model_id="catboost",
        display_name="CatBoost",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "ngboost": ModelSpec(
        model_id="ngboost",
        display_name="NGBoost",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "pgbm": ModelSpec(
        model_id="pgbm",
        display_name="PGBM",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "histgradientboosting": ModelSpec(
        model_id="histgradientboosting",
        display_name="Hist Gradient Boosting",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "randomforest": ModelSpec(
        model_id="randomforest",
        display_name="Random Forest",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "gradientboosting": ModelSpec(
        model_id="gradientboosting",
        display_name="Gradient Boosting",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "gpboost": ModelSpec(
        model_id="gpboost",
        display_name="GPBoost",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "tabnet": ModelSpec(
        model_id="tabnet",
        display_name="TabNet",
        phases=frozenset({PHASE_TUNING, PHASE_QUANTILE, PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
    "hcm": ModelSpec(
        model_id="hcm",
        display_name="Hyperspherical Confidence Mapping",
        phases=frozenset({PHASE_PROBABILISTIC, PHASE_CONFORMAL_STANDARD, PHASE_CONFORMAL_ADVANCED, PHASE_EXPLAINABILITY}),
    ),
}


def list_core_models() -> tuple[str, ...]:
    return CORE_MODELS


def list_supported_models(phase: str) -> list[str]:
    return [model_id for model_id, spec in _MODEL_SPECS.items() if phase in spec.phases]


def validate_models(models: list[str] | tuple[str, ...] | None, phase: str) -> list[str]:
    if models is None:
        return list_supported_models(phase)

    normalized: list[str] = []
    for model_id in models:
        if model_id not in _MODEL_SPECS:
            available = ", ".join(sorted(_MODEL_SPECS))
            raise ValueError(f"Unknown model '{model_id}'. Available models: {available}")
        if phase not in _MODEL_SPECS[model_id].phases:
            raise ValueError(f"Model '{model_id}' does not support phase '{phase}'.")
        normalized.append(model_id)
    return normalized


def _load_lightgbm():
    from lightgbm import LGBMRegressor

    return LGBMRegressor


def _load_xgboost():
    from xgboost import XGBRegressor

    return XGBRegressor


def _load_catboost():
    from catboost import CatBoostRegressor

    return CatBoostRegressor


def _load_ngboost():
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import LogScore

    return NGBRegressor, Normal, LogScore


def _load_pgbm():
    from pgbm.sklearn import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor


def _load_histgradientboosting():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor


def _load_randomforest():
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor


def _load_gradientboosting():
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor


def _load_gpboost():
    from gpboost import GPBoostRegressor

    return GPBoostRegressor


def _load_tabnet():
    from pytorch_tabnet.tab_model import TabNetRegressor

    return TabNetRegressor


def _point_defaults(model_id: str) -> dict[str, Any]:
    if model_id == "lightgbm":
        return {"random_state": 42, "n_estimators": 300}
    if model_id == "xgboost":
        return {
            "random_state": 42,
            "n_estimators": 300,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }
    if model_id == "catboost":
        return {"random_state": 42, "verbose": False, "iterations": 300}
    if model_id == "ngboost":
        return {"random_state": 42, "n_estimators": 400, "verbose": False}
    if model_id == "pgbm":
        return {"random_state": 42, "max_iter": 300}
    if model_id == "histgradientboosting":
        return {"random_state": 42, "max_iter": 300}
    if model_id == "randomforest":
        return {"random_state": 42, "n_estimators": 300, "n_jobs": -1}
    if model_id == "gradientboosting":
        return {"random_state": 42, "n_estimators": 300}
    if model_id == "gpboost":
        return {"random_state": 42, "n_estimators": 300, "verbose": 0}
    if model_id == "tabnet":
        return {"seed": 42, "verbose": 0}
    if model_id == "hcm":
        return {
            "lr": 1e-3,
            "epochs": 200,
            "batch_size": 64,
            "patience": 15,
            "weight_decay": 1e-5,
            "seed": 42,
            "device": "cpu",
        }
    raise ValueError(f"Unsupported model '{model_id}'")


class NGBoostQuantileWrapper:
    def __init__(self, quantile: float, **params):
        self.quantile = quantile
        self.params = params
        self.model = None

    def fit(self, X, y):
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from ngboost.scores import LogScore
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_ = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        fit_params = dict(self.params)
        fit_params.setdefault("Dist", Normal)
        fit_params.setdefault("Score", LogScore)
        self.model = NGBRegressor(**fit_params)
        self.model.fit(X_, y_)
        return self

    def predict(self, X):
        from scipy.stats import norm
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        dist = self.model.pred_dist(X_)
        mu = np.asarray(dist.loc).ravel()
        sigma = np.asarray(dist.scale).ravel()
        z = norm.ppf(self.quantile)
        return mu + z * sigma


class TabNetQuantileWrapper:
    def __init__(self, quantile: float, **params):
        self.quantile = quantile
        self.params = params
        self.model = None

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_ = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        if len(y_.shape) == 1:
            y_ = y_.reshape(-1, 1)

        def quantile_loss(q):
            def loss(y_pred, y_true):
                error = y_true - y_pred
                return torch.mean(torch.maximum(q * error, (q - 1) * error))
            return loss

        fit_params = dict(self.params)
        self.model = TabNetRegressor(**fit_params)
        self.model.fit(
            X_,
            y_,
            eval_set=[(X_, y_)],
            loss_fn=quantile_loss(self.quantile),
            max_epochs=200,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        return self

    def predict(self, X):
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        return self.model.predict(X_).ravel()


class RandomForestQuantileWrapper:
    def __init__(self, quantile: float, **params):
        self.quantile = quantile
        self.params = params
        self.model = None

    def fit(self, X, y):
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_ = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_, y_)
        return self

    def predict(self, X):
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        preds = np.stack([tree.predict(X_) for tree in self.model.estimators_], axis=0)
        return np.quantile(preds, self.quantile, axis=0)


class PGBMQuantileWrapper:
    def __init__(self, quantile: float, **params):
        self.quantile = quantile
        self.params = params
        self.model = None

    def fit(self, X, y):
        from pgbm import PGBM
        import torch
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_ = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        fit_params = dict(self.params)
        for key in ['Dist', 'Score']:
            fit_params.pop(key, None)

        def mseloss_objective(yhat, y_t, sample_weight=None):
            if not torch.is_tensor(yhat):
                yhat = torch.from_numpy(np.array(yhat)).float()
            if not torch.is_tensor(y_t):
                y_t = torch.from_numpy(np.array(y_t)).float()
            return yhat - y_t, torch.ones_like(yhat)

        def rmseloss_metric(yhat, y_t, sample_weight=None):
            if not torch.is_tensor(yhat):
                yhat = torch.from_numpy(np.array(yhat)).float()
            if not torch.is_tensor(y_t):
                y_t = torch.from_numpy(np.array(y_t)).float()
            return torch.sqrt(torch.mean((yhat - y_t) ** 2))

        self.model = PGBM()
        self.model.train(
            train_set=(X_, y_),
            objective=mseloss_objective,
            metric=rmseloss_metric,
            params=fit_params
        )
        return self

    def predict(self, X):
        import numpy as np
        X_ = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        pred_dist = self.model.predict_dist(X_)
        pred_dist_np = pred_dist.cpu().numpy() if hasattr(pred_dist, "cpu") else np.asarray(pred_dist)
        return np.quantile(pred_dist_np, self.quantile, axis=0)


def build_estimator(
    model_id: str,
    phase: str,
    params: dict[str, Any] | None = None,
    quantile: float | None = None,
):
    params = dict(params or {})

    # Raise error if model doesn't support quantile regression but requested
    if phase == PHASE_QUANTILE and model_id not in (
        "lightgbm",
        "xgboost",
        "catboost",
        "gradientboosting",
        "gpboost",
        "histgradientboosting",
        "pgbm",
        "ngboost",
        "tabnet",
        "randomforest",
    ):
        raise ValueError(
            f"Model '{model_id}' does not support quantile regression (PHASE_QUANTILE)."
        )

    if model_id == "lightgbm":
        LGBMRegressor = _load_lightgbm()
        base = _point_defaults(model_id)
        if phase == PHASE_QUANTILE:
            base.update({"objective": "quantile", "alpha": quantile})
        base.update(params)
        return LGBMRegressor(**base)

    if model_id == "xgboost":
        XGBRegressor = _load_xgboost()
        base = _point_defaults(model_id)
        if phase == PHASE_QUANTILE:
            base.update({"objective": "reg:quantileerror", "quantile_alpha": quantile})
        base.update(params)
        return XGBRegressor(**base)

    if model_id == "catboost":
        CatBoostRegressor = _load_catboost()
        base = _point_defaults(model_id)
        if phase == PHASE_QUANTILE:
            base.update({"loss_function": f"Quantile:alpha={quantile}"})
        base.update(params)
        return CatBoostRegressor(**base)

    if model_id == "ngboost":
        base = _point_defaults(model_id)
        base.update(params)
        if phase == PHASE_QUANTILE:
            return NGBoostQuantileWrapper(quantile=quantile, **base)
        NGBRegressor, Normal, LogScore = _load_ngboost()
        base.setdefault("Dist", Normal)
        base.setdefault("Score", LogScore)
        return NGBRegressor(**base)

    if model_id == "pgbm":
        base = _point_defaults(model_id)
        base.update(params)
        if phase == PHASE_QUANTILE:
            return PGBMQuantileWrapper(quantile=quantile, **base)
        PGBMRegressor = _load_pgbm()
        return PGBMRegressor(**base)

    if model_id == "histgradientboosting":
        HistGradientBoostingRegressor = _load_histgradientboosting()
        base = _point_defaults(model_id)
        if phase == PHASE_QUANTILE:
            base.update({"loss": "quantile", "quantile": quantile})
        base.update(params)
        return HistGradientBoostingRegressor(**base)

    if model_id == "randomforest":
        base = _point_defaults(model_id)
        base.update(params)
        if phase == PHASE_QUANTILE:
            return RandomForestQuantileWrapper(quantile=quantile, **base)
        RandomForestRegressor = _load_randomforest()
        return RandomForestRegressor(**base)

    if model_id == "gradientboosting":
        GradientBoostingRegressor = _load_gradientboosting()
        base = _point_defaults(model_id)
        if phase == PHASE_QUANTILE:
            base.update({"loss": "quantile", "alpha": quantile})
        base.update(params)
        return GradientBoostingRegressor(**base)

    if model_id == "gpboost":
        GPBoostRegressor = _load_gpboost()
        base = _point_defaults(model_id)
        if phase == PHASE_QUANTILE:
            base.update({"objective": "quantile", "alpha": quantile})
        base.update(params)
        return GPBoostRegressor(**base)

    if model_id == "tabnet":
        base = _point_defaults(model_id)
        base.update(params)
        if phase == PHASE_QUANTILE:
            return TabNetQuantileWrapper(quantile=quantile, **base)
        TabNetRegressor = _load_tabnet()
        return TabNetRegressor(**base)

    if model_id == "hcm":
        from .probabilistic_advanced import HCMRegressor
        base = _point_defaults(model_id)
        base.update(params)
        return HCMRegressor(**base)

    raise ValueError(f"Unsupported model '{model_id}'")


def suggest_hyperparameters(trial: Any, model_id: str) -> dict[str, Any]:
    if model_id == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    if model_id == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        }
    if model_id == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        }
    if model_id == "ngboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "minibatch_frac": trial.suggest_float("minibatch_frac", 0.5, 1.0),
            "col_sample": trial.suggest_float("col_sample", 0.5, 1.0),
        }
    if model_id == "pgbm":
        return {
            "max_iter": trial.suggest_int("max_iter", 150, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 2.0, log=True),
        }
    if model_id == "randomforest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if model_id == "gradientboosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if model_id == "gpboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        }
    if model_id == "tabnet":
        return {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 10),
            "gamma": trial.suggest_float("gamma", 1.0, 2.0),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
        }
    if model_id == "hcm":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": trial.suggest_int("epochs", 50, 300),
        }
    raise ValueError(f"No search space configured for model '{model_id}'")
