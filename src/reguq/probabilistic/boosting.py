"""Gaussian probabilistic boosting: NGBoost, PGBM and IBUG."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .._base import ArrayLike, as_1d_float
from .._utils import check_fitted, fit_clone, predict_1d
from ..metrics import gaussian_nll
from ..splits import train_calib_split
from ._base import BaseDistributionalRegressor, require
from ._dist import GaussianPrediction, SamplePrediction


class NGBoostRegressor(BaseDistributionalRegressor):
    """NGBoost with a Normal distribution (Duan et al. 2020).

    ``pred_dist(X).params`` gives ``loc`` and ``scale``; ``scale`` is the
    standard deviation of the Normal. Extra keyword arguments go to
    ``ngboost.NGBRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        random_state: int | None = None,
        ngb_params: dict[str, Any] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.random_state = random_state
        self.ngb_params = ngb_params

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_val: ArrayLike | None = None,
        y_val: ArrayLike | None = None,
    ) -> NGBoostRegressor:
        ngboost = require("ngboost", "ngboost")
        from ngboost.distns import Normal

        self.model_ = ngboost.NGBRegressor(
            Dist=Normal,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            random_state=self.random_state,
            verbose=False,
            **(self.ngb_params or {}),
        )
        kw = {} if X_val is None else {"X_val": X_val, "Y_val": as_1d_float(y_val)}
        self.model_.fit(X, as_1d_float(y), **kw)
        return self

    def predict_dist(self, X: ArrayLike) -> GaussianPrediction:
        check_fitted(self, "model_")
        p = self.model_.pred_dist(X).params
        return GaussianPrediction(np.asarray(p["loc"]), np.maximum(np.asarray(p["scale"]), 1e-12))


class PGBMRegressor(BaseDistributionalRegressor):
    """Probabilistic gradient boosting machines (Sprangers et al. 2021).

    Uses ``pgbm.sklearn.HistGradientBoostingRegressor`` whose
    ``predict(X, return_std=True)`` returns mean and **standard deviation**.
    For ``distribution != "normal"`` the predictive distribution is returned
    as ``n_samples`` draws from PGBM's ``sample``.
    """

    def __init__(
        self,
        distribution: str = "normal",
        n_samples: int = 1000,
        random_state: int | None = None,
        pgbm_params: dict[str, Any] | None = None,
    ) -> None:
        self.distribution = distribution
        self.n_samples = n_samples
        self.random_state = random_state
        self.pgbm_params = pgbm_params

    def fit(self, X: ArrayLike, y: ArrayLike) -> PGBMRegressor:
        sk = require("pgbm.sklearn", "pgbm")
        self.model_ = sk.HistGradientBoostingRegressor(
            distribution=self.distribution,
            with_variance=True,
            random_state=self.random_state,
            **(self.pgbm_params or {}),
        )
        self.model_.fit(np.asarray(X, dtype=float), as_1d_float(y))
        return self

    def predict_dist(self, X: ArrayLike) -> GaussianPrediction | SamplePrediction:
        check_fitted(self, "model_")
        mu, std = self.model_.predict(np.asarray(X, dtype=float), return_std=True)
        std = np.maximum(std, 1e-12)
        if self.distribution == "normal":
            return GaussianPrediction(mu, std)
        draws = self.model_.sample(
            mu, std, n_estimates=self.n_samples, random_state=self.random_state or 0
        )
        return SamplePrediction(np.asarray(draws).T)


# ---------------------------------------------------------------- IBUG
def _leaf_matrix(model: Any, X: ArrayLike) -> np.ndarray:
    """Leaf index of every sample in every tree, shape ``(n, n_trees)``."""
    name = type(model).__name__
    if name in ("LGBMRegressor",):
        leaves = model.predict(X, pred_leaf=True)
    elif name in ("CatBoostRegressor",):
        leaves = model.calc_leaf_indexes(X)
    elif hasattr(model, "apply"):  # sklearn forests/GBR, XGBRegressor
        leaves = model.apply(X)
    else:
        raise TypeError(
            f"IBUG needs a tree ensemble exposing leaf indices; {name} is not supported "
            "(use LightGBM, XGBoost, CatBoost, or sklearn RandomForest/ExtraTrees/GradientBoosting)"
        )
    leaves = np.asarray(leaves)
    return leaves.reshape(leaves.shape[0], -1).astype(np.int64)


def _one_hot(leaves: np.ndarray, offsets: np.ndarray, n_cols: int) -> sparse.csr_matrix:
    n, t = leaves.shape
    cols = (leaves + offsets[None, :]).ravel()
    rows = np.repeat(np.arange(n), t)
    return sparse.csr_matrix((np.ones(n * t, dtype=np.float32), (rows, cols)), shape=(n, n_cols))


_K_GRID = (3, 5, 7, 9, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501, 601, 701)
_CAL_BASE = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1e0, 1e1, 1e2, 1e3)
_CAL_MULT = (1.0, 2.5, 5.0)


class IBUGRegressor(BaseDistributionalRegressor):
    """Instance-based uncertainty for gradient-boosted trees (Brophy & Lowd 2022).

    The point model is fitted on the training part. For a test point the
    affinity to training point i is the number of trees in which both land
    in the same leaf; the predictive std is the std of the ``k``
    highest-affinity training **targets**. ``k`` and an additive or
    multiplicative std calibration are selected on a validation split by
    Gaussian NLL, with the grids of the reference implementation
    (``jjbrophy47/ibug``). The returned ``scale`` is a standard deviation -
    it must not be square-rooted again.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        k: int | None = None,
        val_size: float | int = 0.2,
        time_ordered: bool = False,
        variance_calibration: bool = True,
        random_state: int | None = None,
        batch_size: int = 512,
    ) -> None:
        self.estimator = estimator
        self.k = k
        self.val_size = val_size
        self.time_ordered = time_ordered
        self.variance_calibration = variance_calibration
        self.random_state = random_state
        self.batch_size = batch_size

    def _neighbor_std(self, X: ArrayLike, ks: tuple[int, ...]) -> np.ndarray:
        """Std of the top-k neighbours' targets for each k in ``ks``: shape (n, len(ks))."""
        leaves = _leaf_matrix(self.model_, X)
        L = _one_hot(leaves, self.offsets_, self.n_leaf_cols_)
        kmax = max(ks)
        out = np.empty((leaves.shape[0], len(ks)))
        for start in range(0, leaves.shape[0], self.batch_size):
            aff = (L[start : start + self.batch_size] @ self.train_onehot_.T).toarray()
            # top-kmax by affinity (ties broken by training order, stable)
            idx = np.argsort(-aff, axis=1, kind="stable")[:, :kmax]
            vals = self.y_train_[idx]
            for j, k in enumerate(ks):
                out[start : start + self.batch_size, j] = vals[:, :k].std(axis=1)
        return out

    def fit(self, X: ArrayLike, y: ArrayLike) -> IBUGRegressor:
        s = train_calib_split(
            X, y, self.val_size, time_ordered=self.time_ordered, random_state=self.random_state
        )
        self.model_ = fit_clone(self.estimator, s.X_train, s.y_train)
        self.y_train_ = s.y_train
        tr_leaves = _leaf_matrix(self.model_, s.X_train)
        val_leaves = _leaf_matrix(self.model_, s.X_calib)
        n_per_tree = np.maximum(tr_leaves.max(axis=0), val_leaves.max(axis=0)) + 1
        self.offsets_ = np.concatenate([[0], np.cumsum(n_per_tree)[:-1]])
        self.n_leaf_cols_ = int(n_per_tree.sum())
        self.train_onehot_ = _one_hot(tr_leaves, self.offsets_, self.n_leaf_cols_)

        n_tr = s.y_train.size
        ks = (int(self.k),) if self.k is not None else tuple(k for k in _K_GRID if k <= n_tr)
        mu_val = predict_1d(self.model_, s.X_calib)
        std_val = self._neighbor_std(s.X_calib, ks) + 1e-12
        scores = [gaussian_nll(s.y_calib, mu_val, std_val[:, j]) for j in range(len(ks))]
        best = int(np.argmin(scores))
        self.k_ = ks[best]
        sv = std_val[:, best]
        pos = sv[sv > 1e-12]
        self.min_scale_ = float(pos.min()) if pos.size else 1e-12
        sv = np.maximum(sv, self.min_scale_)
        self.gamma_, self.delta_ = 1.0, 0.0
        if self.variance_calibration:
            best_score = np.inf
            for op in ("add", "mult"):
                for b in _CAL_BASE:
                    for m in _CAL_MULT:
                        if op == "mult" and b == 0.0:
                            continue
                        val = b * m
                        cand = sv + val if op == "add" else sv * val
                        if (cand <= 0).any():
                            continue
                        sc = gaussian_nll(s.y_calib, mu_val, cand)
                        if sc < best_score:
                            best_score = sc
                            self.gamma_, self.delta_ = (1.0, val) if op == "add" else (val, 0.0)
        return self

    def predict_dist(self, X: ArrayLike) -> GaussianPrediction:
        check_fitted(self, "k_")
        mu = predict_1d(self.model_, X)
        std = np.maximum(self._neighbor_std(X, (self.k_,))[:, 0], self.min_scale_)
        return GaussianPrediction(mu, std * self.gamma_ + self.delta_)
