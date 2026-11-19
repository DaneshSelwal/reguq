"""Evaluation metrics for point, interval and distributional forecasts."""

from __future__ import annotations

from .._base import ArrayLike, check_alpha
from .distributional import (
    calibration_curve,
    calibration_error,
    crps_gaussian,
    crps_quantile,
    crps_sample,
    ence,
    gaussian_nll,
    pinball_loss,
    pit_gaussian,
    pit_sample,
    quantile_grid_weights,
)
from .intervals import (
    binned_coverage,
    coverage,
    coverage_width_criterion,
    covered,
    interval_score,
    mean_width,
    normalized_mean_width,
    size_stratified_coverage,
)
from .point import bias, mae, r2, rmse


def interval_report(
    y_true: ArrayLike, interval: ArrayLike, alpha: float, *, n_bins: int = 5
) -> dict[str, float]:
    """All interval metrics for one method at miscoverage ``alpha``.

    ``interval`` has shape ``(n, 2)``. Returns coverage (PICP), the target
    ``1 - alpha``, mean width (MPIW), normalized width (PINAW), the Winkler
    interval score, CWC and size-stratified coverage.
    """
    check_alpha(alpha)
    return {
        "target_coverage": 1.0 - alpha,
        "coverage": coverage(y_true, interval),
        "mean_width": mean_width(y_true, interval),
        "normalized_width": normalized_mean_width(y_true, interval),
        "interval_score": interval_score(y_true, interval, alpha=alpha),
        "cwc": coverage_width_criterion(y_true, interval, alpha=alpha),
        "ssc": size_stratified_coverage(y_true, interval, n_bins=n_bins),
    }


def point_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """RMSE, MAE, R^2 and bias of a point prediction."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


__all__ = [
    "bias",
    "binned_coverage",
    "calibration_curve",
    "calibration_error",
    "coverage",
    "coverage_width_criterion",
    "covered",
    "crps_gaussian",
    "crps_quantile",
    "crps_sample",
    "ence",
    "gaussian_nll",
    "interval_report",
    "interval_score",
    "mae",
    "mean_width",
    "normalized_mean_width",
    "pinball_loss",
    "pit_gaussian",
    "pit_sample",
    "point_report",
    "quantile_grid_weights",
    "r2",
    "rmse",
    "size_stratified_coverage",
]
