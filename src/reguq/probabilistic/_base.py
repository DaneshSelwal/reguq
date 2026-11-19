"""Base class for distributional regressors."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from .._base import ArrayLike, BaseIntervalRegressor
from ._dist import GaussianPrediction, SamplePrediction


def require(module: str, extra: str) -> Any:
    """Import an optional dependency or raise an actionable ImportError."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"{module} is required here: pip install 'reguq[{extra}]'") from e


class BaseDistributionalRegressor(BaseIntervalRegressor):
    """A regressor that returns a full predictive distribution.

    Subclasses implement :meth:`fit` and :meth:`predict_dist`; :meth:`predict`
    (the predictive mean) and :meth:`predict_interval` (central interval of
    the predictive distribution) follow. These intervals are **model-based,
    not conformal**: wrap the model in a conformal method (e.g.
    :class:`reguq.conformal.SplitConformalRegressor` with
    ``score="normalized"`` and the predicted std as scale) for a
    finite-sample guarantee.
    """

    @abstractmethod
    def predict_dist(self, X: ArrayLike) -> GaussianPrediction | SamplePrediction: ...

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.predict_dist(X).mean()

    def predict_std(self, X: ArrayLike) -> np.ndarray:
        return self.predict_dist(X).std()

    def predict_interval(self, X: ArrayLike, alpha: float = 0.1) -> np.ndarray:
        return self.predict_dist(X).interval(alpha)


class StdEstimator:
    """Adapter exposing ``predict(X) -> predicted std`` of a fitted distributional model.

    Lets a distributional model serve as the ``scale_estimator`` of a
    normalized split-conformal regressor with ``prefit=True``.
    """

    def __init__(self, model: BaseDistributionalRegressor) -> None:
        self.model = model

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.model.predict_std(X)
