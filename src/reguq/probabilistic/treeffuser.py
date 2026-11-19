"""Treeffuser: diffusion with gradient-boosted score models (Beltran-Velez et al. 2024)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._base import ArrayLike, as_1d_float
from .._utils import check_fitted
from ._base import BaseDistributionalRegressor, require
from ._dist import SamplePrediction


class TreeffuserRegressor(BaseDistributionalRegressor):
    """Wrapper around ``treeffuser.Treeffuser`` returning per-point samples.

    Treeffuser trains its *own* LightGBM score models; do **not** pass the
    hyper-parameters tuned for a point regressor (learning rate, early
    stopping, ...) - that mismatch produced degenerate samples in the v1.2
    notebooks. ``treeffuser_params`` go to ``Treeffuser(...)`` unchanged.
    """

    def __init__(
        self,
        n_samples: int = 200,
        n_steps: int = 50,
        seed: int | None = 0,
        treeffuser_params: dict[str, Any] | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.seed = seed
        self.treeffuser_params = treeffuser_params

    def fit(self, X: ArrayLike, y: ArrayLike) -> TreeffuserRegressor:
        tf = require("treeffuser", "treeffuser")
        self.model_ = tf.Treeffuser(seed=self.seed, **(self.treeffuser_params or {}))
        self.model_.fit(np.asarray(X, dtype=float), as_1d_float(y))
        return self

    def predict_dist(self, X: ArrayLike) -> SamplePrediction:
        check_fitted(self, "model_")
        s = self.model_.sample(
            np.asarray(X, dtype=float),
            n_samples=self.n_samples,
            n_steps=self.n_steps,
            seed=self.seed,
        )  # (n_samples, n, y_dim)
        s = np.asarray(s, dtype=float)
        s = s[..., 0] if s.ndim == 3 else s
        return SamplePrediction(s.T)
