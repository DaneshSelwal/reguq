import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from reguq.reporting import (  # noqa: E402
    plot_binned_coverage,
    plot_calibration,
    plot_intervals,
    save_all_plots,
    write_excel,
)


@pytest.fixture
def results():
    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    df = pd.DataFrame({"y_true": y, "y_pred": y * 0.9, "lower": y - 1, "upper": y + 1})
    metrics = pd.DataFrame(
        [
            {
                "method": "split",
                "model": "m",
                "status": "ok",
                "coverage": 1.0,
                "normalized_width": 0.3,
            }
        ]
    )
    return metrics, {"split__m": df}


def test_plots_return_figures(results):
    _, preds = results
    d = preds["split__m"]
    assert plot_intervals(d.y_true, d.y_pred, d.lower, d.upper) is not None
    assert plot_calibration(np.random.default_rng(1).uniform(size=200)) is not None
    assert plot_binned_coverage(d.y_true, d.lower, d.upper, alpha=0.1) is not None


def test_save_all_plots(tmp_path, results):
    metrics, preds = results
    paths = save_all_plots(metrics, preds, tmp_path, alpha=0.1)
    assert all(p.exists() for p in paths) and len(paths) == 2


def test_write_excel_sheet_names(tmp_path, results):
    pytest.importorskip("openpyxl")
    metrics, preds = results
    long = {"x" * 40: preds["split__m"], "x" * 41: preds["split__m"]}
    path = write_excel(metrics, {**preds, **long}, tmp_path / "r.xlsx")
    sheets = pd.ExcelFile(path).sheet_names
    assert sheets[0] == "metrics" and len(set(sheets)) == 4 and all(len(s) <= 31 for s in sheets)
