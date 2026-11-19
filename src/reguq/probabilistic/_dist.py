"""Predictive-distribution containers returned by ``predict_dist``.

Two families cover every probabilistic model in reguq:

* :class:`GaussianPrediction` - per-point mean and **standard deviation**;
* :class:`SamplePrediction` - per-point draws, shape ``(n, m)``.

Both expose the same methods, so evaluation code never needs to know which
model produced the forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .._base import ArrayLike, as_1d_float, as_interval, check_alpha
from ..metrics import crps_gaussian, crps_sample, gaussian_nll, pit_gaussian, pit_sample


@dataclass(frozen=True)
class GaussianPrediction:
    """Independent Gaussian forecasts :math:`N(\\mu_i, \\sigma_i^2)`; ``sigma`` is a std."""

    mu: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        mu = as_1d_float(self.mu, "mu")
        sigma = np.broadcast_to(np.asarray(self.sigma, float), mu.shape).copy()
        if (sigma <= 0).any() or not np.isfinite(sigma).all():
            raise ValueError("sigma must be finite and > 0")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)

    def mean(self) -> np.ndarray:
        return self.mu

    def std(self) -> np.ndarray:
        return self.sigma

    def quantiles(self, taus: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(taus, float))
        return self.mu[:, None] + self.sigma[:, None] * norm.ppf(t)[None, :]

    def interval(self, alpha: float = 0.1) -> np.ndarray:
        z = norm.ppf(1 - check_alpha(alpha) / 2)
        return as_interval(self.mu - z * self.sigma, self.mu + z * self.sigma)

    def sample(self, m: int, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        return self.mu[:, None] + self.sigma[:, None] * rng.standard_normal((self.mu.size, m))

    def crps(self, y: ArrayLike) -> float:
        return float(crps_gaussian(y, self.mu, self.sigma))

    def nll(self, y: ArrayLike) -> float:
        return float(gaussian_nll(y, self.mu, self.sigma))

    def pit(self, y: ArrayLike) -> np.ndarray:
        return pit_gaussian(y, self.mu, self.sigma)


@dataclass(frozen=True)
class SamplePrediction:
    """Empirical forecasts from ``m`` draws per point; ``samples`` has shape ``(n, m)``."""

    samples: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.samples, dtype=float)
        if s.ndim != 2 or s.shape[1] < 2:
            raise ValueError(f"samples must have shape (n, m>=2); got {s.shape}")
        object.__setattr__(self, "samples", s)

    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=1)

    def std(self) -> np.ndarray:
        return self.samples.std(axis=1, ddof=1)

    def quantiles(self, taus: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(taus, float))
        return np.quantile(self.samples, t, axis=1).T

    def interval(self, alpha: float = 0.1) -> np.ndarray:
        a = check_alpha(alpha)
        q = self.quantiles([a / 2, 1 - a / 2])
        return as_interval(q[:, 0], q[:, 1])

    def sample(self, m: int, random_state: int | None = None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, self.samples.shape[1], size=(self.samples.shape[0], m))
        return np.take_along_axis(self.samples, idx, axis=1)

    def crps(self, y: ArrayLike) -> float:
        return float(crps_sample(y, self.samples, fair=True))

    def nll(self, y: ArrayLike) -> float:
        """Gaussian NLL using the sample mean and std (a moment approximation)."""
        return float(gaussian_nll(y, self.mean(), np.maximum(self.std(), 1e-12)))

    def pit(self, y: ArrayLike, random_state: int | None = 0) -> np.ndarray:
        return pit_sample(y, self.samples, random_state=random_state)


Prediction = GaussianPrediction | SamplePrediction
