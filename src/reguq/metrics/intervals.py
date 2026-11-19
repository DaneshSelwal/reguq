r"""Prediction-interval metrics.

Every function accepts either ``(y_true, interval)`` with ``interval`` of
shape ``(n, 2)``, or ``(y_true, lower, upper)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._base import ArrayLike, as_1d_float, check_alpha, check_interval


def _bounds(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = as_1d_float(y_true, "y_true")
    if upper is None:
        lo, hi = check_interval(lower)
    else:
        lo, hi = as_1d_float(lower, "lower"), as_1d_float(upper, "upper")
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(f"shape mismatch: y {y.shape}, lower {lo.shape}, upper {hi.shape}")
    return y, lo, hi


def covered(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike | None = None) -> np.ndarray:
    """Boolean vector ``lower_i <= y_i <= upper_i`` (element-wise, no broadcasting)."""
    y, lo, hi = _bounds(y_true, lower, upper)
    return (y >= lo) & (y <= hi)


def coverage(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike | None = None) -> float:
    """Empirical coverage, a.k.a. PICP: fraction of ``y`` inside its interval."""
    return float(covered(y_true, lower, upper).mean())


def mean_width(y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike | None = None) -> float:
    """Mean interval width (MPIW). Infinite if any interval is unbounded."""
    _, lo, hi = _bounds(y_true, lower, upper)
    return float(np.mean(hi - lo))


def normalized_mean_width(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike | None = None
) -> float:
    """MPIW divided by the range of ``y_true`` (PINAW, Khosravi et al. 2011)."""
    y, lo, hi = _bounds(y_true, lower, upper)
    rng = float(np.ptp(y))
    if rng == 0:
        raise ValueError("y_true has zero range; normalized width undefined")
    return float(np.mean(hi - lo) / rng)


def interval_score(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike | None = None,
    *,
    alpha: float,
    reduction: str = "mean",
) -> float | np.ndarray:
    r"""Winkler / interval score (Gneiting & Raftery 2007, eq. 43). Lower is better.

    .. math::
        IS_\alpha(l, u; y) = (u - l) + \tfrac{2}{\alpha}(l - y)\mathbb 1\{y < l\}
                                     + \tfrac{2}{\alpha}(y - u)\mathbb 1\{y > u\}

    It is a proper scoring rule for the central ``1 - alpha`` interval.
    """
    check_alpha(alpha)
    y, lo, hi = _bounds(y_true, lower, upper)
    s = (hi - lo) + (2.0 / alpha) * np.maximum(lo - y, 0) + (2.0 / alpha) * np.maximum(y - hi, 0)
    return float(s.mean()) if reduction == "mean" else s


def coverage_width_criterion(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike | None = None,
    *,
    alpha: float,
    eta: float = 50.0,
) -> float:
    r"""Coverage-width criterion (Khosravi, Nahavandi & Creighton 2011).

    .. math::
        CWC = PINAW\,\big(1 + \gamma\, e^{-\eta (PICP - \mu)}\big),\quad
        \mu = 1-\alpha,\; \gamma = \mathbb 1\{PICP < \mu\}

    Note: CWC is not a proper scoring rule; prefer :func:`interval_score`
    for ranking methods. It is provided for comparability with the literature.
    """
    check_alpha(alpha)
    picp = coverage(y_true, lower, upper)
    pinaw = normalized_mean_width(y_true, lower, upper)
    mu = 1.0 - alpha
    gamma = 1.0 if picp < mu else 0.0
    return float(pinaw * (1.0 + gamma * np.exp(-eta * (picp - mu))))


def binned_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike | None = None,
    *,
    by: ArrayLike | None = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Coverage within quantile bins of ``by`` (default: the interval width).

    Binning by width gives size-stratified coverage (Angelopoulos & Bates
    2023, Sec. 3); binning by a feature or by ``y_hat`` gives a check of
    conditional coverage. Returns one row per bin with ``n``, ``coverage``
    and ``mean_width``.
    """
    y, lo, hi = _bounds(y_true, lower, upper)
    width = hi - lo
    key = width if by is None else as_1d_float(by, "by")
    if key.shape != y.shape:
        raise ValueError("`by` must have one value per sample")
    bins = pd.qcut(key, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"bin": bins, "covered": (y >= lo) & (y <= hi), "width": width})
    out = df.groupby("bin", observed=True).agg(
        n=("covered", "size"), coverage=("covered", "mean"), mean_width=("width", "mean")
    )
    return out.reset_index()


def size_stratified_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike | None = None,
    *,
    n_bins: int = 5,
) -> float:
    """Worst-bin coverage across width-quantile bins (SSC). Close to ``1-alpha`` is good."""
    return float(binned_coverage(y_true, lower, upper, n_bins=n_bins)["coverage"].min())
