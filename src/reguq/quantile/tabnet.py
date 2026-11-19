"""Scikit-learn compatible TabNet regressor with point and quantile objectives.

Fixes two v1.2 issues: (i) pytorch-tabnet needs numpy inputs and a 2-D
target, and (ii) early stopping must use a *held-out* validation split with
the *same* loss as training (pinball for quantiles), not training-set MSE.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .._base import ArrayLike, as_1d_float
from .._utils import check_fitted
from ..splits import split_indices


class TabNetRegressor(RegressorMixin, BaseEstimator):
    """TabNet (Arik & Pfister 2021) regressor.

    Parameters
    ----------
    quantile
        ``None`` for squared error, or a level in (0, 1) for the pinball loss.
    val_size
        Fraction of training rows held out for early stopping.
    tabnet_params
        Passed to ``pytorch_tabnet.tab_model.TabNetRegressor``.
    """

    def __init__(
        self,
        quantile: float | None = None,
        *,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        val_size: float = 0.15,
        time_ordered: bool = False,
        random_state: int = 0,
        tabnet_params: dict[str, Any] | None = None,
    ) -> None:
        self.quantile = quantile
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_size = val_size
        self.time_ordered = time_ordered
        self.random_state = random_state
        self.tabnet_params = tabnet_params

    def set_quantile(self, q: float) -> TabNetRegressor:
        """Unfitted copy configured for quantile ``q`` (used by :func:`with_quantile`)."""
        est = clone(self)
        est.quantile = float(q)
        return est

    def fit(self, X: ArrayLike, y: ArrayLike) -> TabNetRegressor:
        from ..probabilistic._base import require

        require("pytorch_tabnet", "tabnet")
        import torch
        from pytorch_tabnet.metrics import Metric
        from pytorch_tabnet.tab_model import TabNetRegressor as _TabNet

        Xa = np.asarray(X, dtype=np.float32)
        ya = as_1d_float(y).astype(np.float32).reshape(-1, 1)
        tr, va = split_indices(
            len(Xa), self.val_size, time_ordered=self.time_ordered, random_state=self.random_state
        )
        q = self.quantile
        kwargs: dict[str, Any] = {}
        if q is not None:
            if not 0 < q < 1:
                raise ValueError("quantile must lie in (0, 1)")
            qv: float = q

            def pinball_torch(y_pred: Any, y_true: Any) -> Any:
                e = y_true - y_pred
                return torch.mean(torch.maximum(qv * e, (qv - 1) * e))

            class Pinball(Metric):
                _name = "pinball"
                _maximize = False

                def __call__(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
                    e = y_true - y_score
                    return float(np.mean(np.maximum(qv * e, (qv - 1) * e)))

            kwargs = {"loss_fn": pinball_torch, "eval_metric": [Pinball]}
        self.model_ = _TabNet(seed=self.random_state, verbose=0, **(self.tabnet_params or {}))
        self.model_.fit(
            Xa[tr],
            ya[tr],
            eval_set=[(Xa[va], ya[va])],
            eval_name=["val"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=min(128, self.batch_size),
            **kwargs,
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_fitted(self, "model_")
        return np.asarray(self.model_.predict(np.asarray(X, dtype=np.float32)), dtype=float).ravel()
