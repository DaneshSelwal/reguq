"""Data loading for the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def load_data(cfg: PipelineConfig) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Read train/test CSVs, strip column-name whitespace, align features."""
    tr = pd.read_csv(cfg.train_path)
    te = pd.read_csv(cfg.test_path)
    tr.columns = [str(c).strip() for c in tr.columns]
    te.columns = [str(c).strip() for c in te.columns]
    target = (cfg.target or tr.columns[-1]).strip()
    test_target = (cfg.test_target or target).strip()
    if target not in tr.columns:
        raise KeyError(f"target {target!r} not in training columns {list(tr.columns)}")
    if test_target not in te.columns:
        raise KeyError(
            f"target {test_target!r} not in test columns {list(te.columns)}. If the test file "
            "uses a different target name, set `test_target` explicitly (and note that the "
            "test then measures a different quantity)."
        )
    feats = cfg.features or [c for c in tr.columns if c != target]
    missing = [c for c in feats if c not in te.columns]
    if missing:
        raise KeyError(f"test file lacks features {missing}")
    return tr[feats], tr[target].to_numpy(float), te[feats], te[test_target].to_numpy(float)
