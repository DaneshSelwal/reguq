"""Metric utilities shared across phases."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def interval_metrics(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> dict[str, float]:
    coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
    width = np.mean(y_upper - y_lower)
    return {
        "coverage": float(coverage),
        "avg_interval_width": float(width),
    }


def gaussian_nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    safe_std = np.maximum(std, 1e-8)
    nll = 0.5 * np.log(2 * math.pi * safe_std**2) + 0.5 * ((y_true - mean) / safe_std) ** 2
    return float(np.mean(nll))


def gaussian_crps(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    safe_std = np.maximum(std, 1e-8)
    z = (y_true - mean) / safe_std
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    crps = safe_std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def to_metrics_frame(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
