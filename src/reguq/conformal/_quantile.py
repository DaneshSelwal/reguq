r"""Finite-sample conformal quantiles.

All conformal methods in reguq reduce to one of these functions, so the
finite-sample correction lives in exactly one place.

Unweighted (exchangeable) case
------------------------------
Given calibration scores :math:`S_1,\dots,S_n` and a test score
:math:`S_{n+1}` that are exchangeable,

.. math::
    \hat q = S_{(k)}, \qquad k = \lceil (n+1)(1-\alpha) \rceil

satisfies :math:`1-\alpha \le P(S_{n+1} \le \hat q) \le 1-\alpha + 1/(n+1)`
(the upper bound when scores are a.s. distinct). If :math:`k > n` the
quantile is :math:`+\infty` (Vovk et al. 2005; Lei et al. 2018; Angelopoulos
& Bates 2023, "A Gentle Introduction to Conformal Prediction", Thm. 1).

Weighted (non-exchangeable) case
--------------------------------
Barber, Candès, Ramdas & Tibshirani (2023), "Conformal prediction beyond
exchangeability", eq. (5): with fixed weights :math:`w_i \in [0, 1]` and the
test point's weight :math:`w_{n+1}` (= 1 by default),

.. math::
    \tilde w_i = \frac{w_i}{\sum_{j=1}^n w_j + w_{n+1}}, \qquad
    \hat q = Q_{1-\alpha}\Big(\sum_{i=1}^n \tilde w_i\,\delta_{S_i}
             + \tilde w_{n+1}\,\delta_{+\infty}\Big),

i.e. the smallest calibration score whose cumulative normalized weight
reaches :math:`1-\alpha`; :math:`+\infty` if the calibration mass never
does. No interpolation is used: the quantile is a step function.
"""

from __future__ import annotations

import math

import numpy as np

from .._base import ArrayLike, as_1d_float, check_alpha


def _validate_scores(scores: ArrayLike) -> np.ndarray:
    s = as_1d_float(scores, "scores")
    if s.size == 0:
        raise ValueError("need at least one calibration score")
    if np.isnan(s).any():
        raise ValueError("calibration scores contain NaN")
    return s


def conformal_rank(n: int, alpha: float) -> int:
    """Return :math:`k = \\lceil (n+1)(1-\\alpha) \\rceil` (1-based order statistic)."""
    check_alpha(alpha)
    # Guard against floating error such as (n+1)*(1-alpha) = 90.00000000001.
    return int(math.ceil((n + 1) * (1.0 - alpha) - 1e-10))


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Upper finite-sample conformal quantile of ``scores`` at level ``1 - alpha``.

    Returns ``np.inf`` when ``n`` is too small, i.e. when
    ``ceil((n+1)(1-alpha)) > n``.
    """
    s = _validate_scores(scores)
    k = conformal_rank(s.size, alpha)
    if k > s.size:
        return float(np.inf)
    return float(np.partition(s, k - 1)[k - 1])


def conformal_lower_quantile(values: ArrayLike, alpha: float) -> float:
    """Lower finite-sample quantile: the :math:`\\lfloor \\alpha(n+1) \\rfloor`-th smallest value.

    This is the lower end used by Jackknife+ / CV+ (Barber et al. 2021,
    "Predictive inference with the jackknife+", eq. 3). Returns ``-np.inf``
    when the rank is below 1.
    """
    v = _validate_scores(values)
    check_alpha(alpha)
    k = int(math.floor(alpha * (v.size + 1) + 1e-10))
    if k < 1:
        return float(-np.inf)
    return float(np.partition(v, k - 1)[k - 1])


def conformal_upper_quantile(values: ArrayLike, alpha: float) -> float:
    """Upper end used by Jackknife+ / CV+: the ``ceil((1-alpha)(n+1))``-th smallest value."""
    return conformal_quantile(values, alpha)


def weighted_conformal_quantile(
    scores: ArrayLike,
    weights: ArrayLike,
    alpha: float,
    test_weight: float = 1.0,
) -> float:
    """Weighted conformal quantile with the test point's mass at :math:`+\\infty`.

    Parameters
    ----------
    scores, weights
        Calibration scores and their (unnormalised, non-negative) weights.
    alpha
        Target miscoverage.
    test_weight
        Weight :math:`w_{n+1}` of the test point (1 in Barber et al. 2023).
    """
    s = _validate_scores(scores)
    w = as_1d_float(weights, "weights")
    if w.shape != s.shape:
        raise ValueError(f"weights {w.shape} must match scores {s.shape}")
    if (w < 0).any() or test_weight < 0:
        raise ValueError("weights must be non-negative")
    check_alpha(alpha)
    total = w.sum() + float(test_weight)
    if total <= 0:
        raise ValueError("total weight must be positive")
    order = np.argsort(s, kind="stable")
    cum = np.cumsum(w[order]) / total
    # Smallest score whose cumulative weight reaches 1 - alpha.
    idx = int(np.searchsorted(cum, 1.0 - alpha - 1e-12, side="left"))
    if idx >= s.size:
        return float(np.inf)
    return float(s[order][idx])
