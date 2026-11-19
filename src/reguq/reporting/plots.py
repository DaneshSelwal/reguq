"""Diagnostic figures (matplotlib, ``report`` extra). Every function returns the Figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._base import ArrayLike, as_1d_float
from ..metrics import binned_coverage, calibration_curve


def _plt() -> Any:
    from ..probabilistic._base import require

    require("matplotlib", "report")
    import matplotlib.pyplot as plt

    return plt


def plot_intervals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    title: str = "",
    sort: bool = False,
    max_points: int = 500,
) -> Any:
    """Prediction intervals over the test index; uncovered points are marked in red.

    ``sort=True`` orders points by prediction (useful for i.i.d. data); keep
    ``False`` for time series so the x-axis is time.
    """
    plt = _plt()
    y, p, lo, hi = (as_1d_float(a) for a in (y_true, y_pred, lower, upper))
    idx = np.argsort(p) if sort else np.arange(y.size)
    idx = idx[:max_points]
    x = np.arange(idx.size)
    miss = (y[idx] < lo[idx]) | (y[idx] > hi[idx])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(x, lo[idx], hi[idx], alpha=0.3, label="interval", step="mid")
    ax.plot(x, p[idx], lw=1, label="prediction")
    ax.scatter(x[~miss], y[idx][~miss], s=8, c="k", label="observed (covered)")
    ax.scatter(
        x[miss], y[idx][miss], s=14, c="tab:red", label=f"observed (missed, {miss.mean():.1%})"
    )
    ax.set_xlabel("sorted by prediction" if sort else "test index")
    ax.set_ylabel("target")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_calibration(pit: ArrayLike, *, title: str = "PIT reliability") -> Any:
    """Reliability diagram: observed vs nominal ``P(PIT <= p)``."""
    plt = _plt()
    p, obs = calibration_curve(pit)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(p, obs, "o-", label="model")
    ax.set_xlabel("nominal level")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_coverage_width(metrics: pd.DataFrame, *, alpha: float) -> Any:
    """Coverage vs normalized width for every successful (method, model) row."""
    plt = _plt()
    ok = metrics[metrics.get("status", "ok") == "ok"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, g in ok.groupby("method"):
        ax.scatter(g["normalized_width"], g["coverage"], label=method, s=28)
    ax.axhline(1 - alpha, color="k", ls="--", lw=1, label=f"target {1 - alpha:.0%}")
    ax.set_xlabel("normalized mean width (PINAW)")
    ax.set_ylabel("test coverage (PICP)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig


def plot_binned_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    alpha: float,
    by: ArrayLike | None = None,
) -> Any:
    """Coverage per width (or ``by``) quantile bin - a check of conditional coverage."""
    plt = _plt()
    tab = binned_coverage(y_true, lower, upper, by=by)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(tab)), tab["coverage"], tick_label=[str(b) for b in tab["bin"]])
    ax.axhline(1 - alpha, color="k", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    return fig


def save_all_plots(
    metrics: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    out_dir: str | Path,
    *,
    alpha: float,
) -> list[Path]:
    """Write the overview and one interval plot per prediction table; returns the paths."""
    plt = _plt()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    fig = plot_coverage_width(metrics, alpha=alpha)
    paths.append(out / "coverage_vs_width.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    for key, df in predictions.items():
        fig = plot_intervals(df["y_true"], df["y_pred"], df["lower"], df["upper"], title=key)
        paths.append(out / f"{key}.png")
        fig.savefig(paths[-1], dpi=120)
        plt.close(fig)
    return paths
