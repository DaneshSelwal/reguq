"""Preprocessing helpers used across phases."""

from __future__ import annotations

import pandas as pd


def coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric when possible and keep original for non-convertible columns."""
    converted = frame.copy()
    for col in converted.columns:
        try:
            converted[col] = pd.to_numeric(converted[col])
        except Exception:
            pass
    return converted
