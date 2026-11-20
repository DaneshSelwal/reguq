"""Excel export (``report`` extra)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _sheet_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[:31] or "sheet"
    cand, i = base, 1
    while cand in used:
        suffix = f"~{i}"
        cand = base[: 31 - len(suffix)] + suffix
        i += 1
    used.add(cand)
    return cand


def write_excel(
    metrics: pd.DataFrame, predictions: dict[str, pd.DataFrame], path: str | Path
) -> Path:
    """One ``metrics`` sheet plus one sheet per prediction table (names made Excel-safe)."""
    from ..probabilistic._base import require

    require("openpyxl", "report")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        metrics.drop(columns=["traceback"], errors="ignore").to_excel(
            xw, sheet_name=_sheet_name("metrics", used), index=False
        )
        for key, df in predictions.items():
            df.to_excel(xw, sheet_name=_sheet_name(key, used), index=False)
    return path
