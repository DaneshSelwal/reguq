r"""Metrics for quantile and full predictive-distribution forecasts."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .._base import ArrayLike, as_1d_float


def _reduce(x: np.ndarray, reduction: str) -> float | np.ndarray:
    if reduction == "mean":
        return float(np.mean(x))
    if reduction == "none":
        return x
    raise ValueError("reduction must be 'mean' or 'none'")


def _check_sigma(sigma: ArrayLike, n: int) -> np.ndarray:
    s = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).astype(float)
    if (s <= 0).any() or not np.isfinite(s).all():
        raise ValueError("sigma must be finite and strictly positive")
    return s


def pinball_loss(
    y_true: ArrayLike, y_quantile: ArrayLike, tau: float, *, reduction: str = "mean"
) -> float | np.ndarray:
    r"""Quantile (pinball) loss :math:`\rho_\tau(y - q) = \max(\tau e, (\tau-1) e)`, ``e = y - q``."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie in (0, 1)")
    y = as_1d_float(y_true)
    q = as_1d_float(y_quantile, "y_quantile")
    e = y - q
    return _reduce(np.maximum(tau * e, (tau - 1.0) * e), reduction)


def quantile_grid_weights(taus: ArrayLike) -> np.ndarray:
    """Weights of each quantile level for a rectangle-rule integral over (0, 1).

    Level :math:`\\tau_k` owns the cell between the midpoints to its
    neighbours, with the outer cells extended to 0 and 1. For the uniform
    midpoint grid :math:`\\tau_k=(k-1/2)/K` every weight equals ``1/K``.
    """
    t = as_1d_float(taus, "taus")
    if (np.diff(t) <= 0).any() or t[0] <= 0 or t[-1] >= 1:
        raise ValueError("taus must be strictly increasing inside (0, 1)")
    edges = np.concatenate([[0.0], (t[1:] + t[:-1]) / 2.0, [1.0]])
    return np.diff(edges)


def crps_quantile(
    y_true: ArrayLike,
    quantile_preds: ArrayLike,
    taus: ArrayLike,
    *,
    reduction: str = "mean",
) -> float | np.ndarray:
    r"""CRPS approximated from predicted quantiles.

    Uses :math:`\mathrm{CRPS}(F, y) = 2\int_0^1 \rho_\tau(y - F^{-1}(\tau))\,d\tau`
    (Gneiting & Ranjan 2011) with :func:`quantile_grid_weights`. Unequal
    spacing is handled by the weights, unlike treating quantiles as an
    equally-weighted ensemble. Accuracy improves with a dense, uniform grid
    (e.g. ``np.arange(1, 100) / 100``).
    """
    y = as_1d_float(y_true)
    q = np.asarray(quantile_preds, dtype=float)
    t = as_1d_float(taus, "taus")
    if q.shape != (y.size, t.size):
        raise ValueError(f"quantile_preds must have shape {(y.size, t.size)}; got {q.shape}")
    w = quantile_grid_weights(t)
    e = y[:, None] - q
    loss = np.maximum(t * e, (t - 1.0) * e)
    return _reduce(2.0 * loss @ w, reduction)


def crps_gaussian(
    y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike, *, reduction: str = "mean"
) -> float | np.ndarray:
    r"""Closed-form CRPS of :math:`N(\mu, \sigma^2)` (Gneiting et al. 2005).

    .. math:: \sigma\big[z(2\Phi(z)-1) + 2\varphi(z) - 1/\sqrt\pi\big],\; z=(y-\mu)/\sigma
    """
    y = as_1d_float(y_true)
    m = np.broadcast_to(np.asarray(mu, dtype=float), y.shape)
    s = _check_sigma(sigma, y.size)
    z = (y - m) / s
    c = s * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return _reduce(c, reduction)


def crps_sample(
    y_true: ArrayLike, samples: ArrayLike, *, fair: bool = False, reduction: str = "mean"
) -> float | np.ndarray:
    r"""CRPS of an empirical predictive distribution given by samples.

    ``samples`` has shape ``(n, m)``: ``m`` draws for each of ``n`` points.
    Uses :math:`E|X-y| - \tfrac12 E|X-X'|`, with the pairwise term computed
    in :math:`O(m \log m)` from order statistics. ``fair=True`` uses the
    unbiased ``m(m-1)`` normalisation (Ferro 2014).
    """
    y = as_1d_float(y_true)
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError(f"samples must have shape (n={y.size}, m); got {x.shape}")
    m = x.shape[1]
    if fair and m < 2:
        raise ValueError("fair CRPS needs at least 2 samples")
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    xs = np.sort(x, axis=1)
    i = np.arange(1, m + 1)
    pair_sum = 2.0 * (xs * (2 * i - m - 1)).sum(axis=1)  # sum_{j,k} |x_j - x_k|
    denom = m * (m - 1) if fair else m * m
    return _reduce(term1 - 0.5 * pair_sum / denom, reduction)


def gaussian_nll(
    y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike, *, reduction: str = "mean"
) -> float | np.ndarray:
    r"""Negative log-likelihood of :math:`N(\mu, \sigma^2)`; ``sigma`` is a standard deviation."""
    y = as_1d_float(y_true)
    m = np.broadcast_to(np.asarray(mu, dtype=float), y.shape)
    s = _check_sigma(sigma, y.size)
    nll = 0.5 * np.log(2.0 * np.pi * s**2) + 0.5 * ((y - m) / s) ** 2
    return _reduce(nll, reduction)


def pit_gaussian(y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike) -> np.ndarray:
    """Probability integral transform :math:`F_i(y_i)` for Gaussian forecasts."""
    y = as_1d_float(y_true)
    return norm.cdf(
        y, loc=np.broadcast_to(np.asarray(mu, float), y.shape), scale=_check_sigma(sigma, y.size)
    )


def pit_sample(
    y_true: ArrayLike, samples: ArrayLike, *, random_state: int | None = None
) -> np.ndarray:
    """Randomised PIT from per-point samples (ties are broken uniformly)."""
    y = as_1d_float(y_true)
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError(f"samples must have shape (n={y.size}, m)")
    below = (x < y[:, None]).mean(axis=1)
    equal = (x == y[:, None]).mean(axis=1)
    u = np.random.default_rng(random_state).uniform(size=y.size)
    return below + u * equal


def calibration_curve(
    pit: ArrayLike, levels: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Observed frequency ``P(PIT <= p)`` at nominal levels ``p``.

    Default levels are ``0.05, 0.10, ..., 0.95``; the trivial endpoints 0 and
    1 (always exact) are excluded so they do not dilute calibration errors.
    """
    u = as_1d_float(pit, "pit")
    p = np.arange(1, 20) / 20 if levels is None else as_1d_float(levels, "levels")
    observed = (u[:, None] <= p[None, :]).mean(axis=0)
    return p, observed


def calibration_error(pit: ArrayLike, levels: ArrayLike | None = None) -> dict[str, float]:
    """Mean-absolute, root-mean-square and max gap of :func:`calibration_curve`."""
    p, obs = calibration_curve(pit, levels)
    gap = obs - p
    return {
        "mace": float(np.mean(np.abs(gap))),
        "rmsce": float(np.sqrt(np.mean(gap**2))),
        "mce": float(np.max(np.abs(gap))),
    }


def ence(y_true: ArrayLike, mu: ArrayLike, sigma: ArrayLike, *, n_bins: int = 10) -> float:
    r"""Expected normalized calibration error (Levi et al. 2022).

    Sorts points by predicted ``sigma`` into ``n_bins`` equal-count bins and
    averages :math:`|RMV_b - RMSE_b| / RMV_b`, where
    :math:`RMV_b=\sqrt{\mathrm{mean}(\sigma^2)}` and RMSE is of ``y - mu``.
    Compares like with like (root-mean-square vs root-mean-square), unlike
    comparing mean sigma to mean absolute error.
    """
    y = as_1d_float(y_true)
    m = np.broadcast_to(np.asarray(mu, float), y.shape)
    s = _check_sigma(sigma, y.size)
    order = np.argsort(s, kind="stable")
    vals = []
    for idx in np.array_split(order, n_bins):
        if idx.size == 0:
            continue
        rmv = np.sqrt(np.mean(s[idx] ** 2))
        rmse = np.sqrt(np.mean((y[idx] - m[idx]) ** 2))
        vals.append(abs(rmv - rmse) / rmv)
    return float(np.mean(vals))
