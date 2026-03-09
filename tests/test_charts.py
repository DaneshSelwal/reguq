from __future__ import annotations

from pathlib import Path

import pandas as pd

from reguq.charts import generate_phase_charts
from reguq.constants import PHASE_QUANTILE
from reguq.types import OutputConfig, PhaseResult


def test_generate_phase_charts_returns_paths_and_sheet_map(tmp_path: Path):
    pred_df = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.1, 1.9, 2.8, 4.2],
            "y_lower": [0.8, 1.5, 2.4, 3.7],
            "y_upper": [1.3, 2.3, 3.2, 4.6],
        }
    )
    metrics_df = pd.DataFrame(
        {
            "model": ["lightgbm"],
            "rmse": [0.2],
            "coverage": [1.0],
            "avg_interval_width": [0.7],
        }
    )
    phase_result = PhaseResult(
        phase=PHASE_QUANTILE,
        predictions={"lightgbm": pred_df},
        metrics=metrics_df,
        params={},
    )
    output_cfg = OutputConfig(
        output_dir=tmp_path,
        export_plots=True,
        embed_excel_charts=True,
        show_inline_plots=False,
    )

    chart_result = generate_phase_charts(
        phase_result=phase_result,
        phase_name=PHASE_QUANTILE,
        output_cfg=output_cfg,
        output_dir=tmp_path,
    )

    assert chart_result.image_paths
    assert "pred_lightgbm" in chart_result.images_by_sheet
    assert "metrics" in chart_result.images_by_sheet
    assert all(path.exists() for path in chart_result.image_paths)
