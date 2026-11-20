"""Figures and Excel export for results (needs the ``report`` extra)."""

from __future__ import annotations

from .excel import write_excel
from .plots import (
    plot_binned_coverage,
    plot_calibration,
    plot_coverage_width,
    plot_intervals,
    save_all_plots,
)

__all__ = [
    "plot_binned_coverage",
    "plot_calibration",
    "plot_coverage_width",
    "plot_intervals",
    "save_all_plots",
    "write_excel",
]
