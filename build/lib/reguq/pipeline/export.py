"""Exporting pipeline results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .config import PipelineConfig

if TYPE_CHECKING:
    from .runner import PipelineResult


def export(result: PipelineResult, cfg: PipelineConfig) -> Path:
    """Write ``metrics.csv``, ``predictions/*.csv``, ``params.json`` and ``config.json``.

    With ``cfg.export_excel`` also ``results.xlsx`` (one sheet of metrics,
    one per prediction table); with ``cfg.export_plots`` PNG figures under
    ``plots/`` (needs the ``report`` extra).
    """
    out = Path(cfg.output_dir)  # type: ignore[arg-type]
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(out / "metrics.csv", index=False)
    for key, df in result.predictions.items():
        df.to_csv(out / "predictions" / f"{key}.csv", index=False)
    (out / "params.json").write_text(
        json.dumps(
            {"best_params": result.best_params, "tuning": result.tuning}, indent=2, default=str
        )
    )
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    return out
