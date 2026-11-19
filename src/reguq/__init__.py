"""reguq: regression uncertainty quantification with tested guarantees.

The top-level import only depends on numpy / scipy / pandas / scikit-learn.
Backends such as torch, LightGBM or MAPIE are imported lazily, inside the
estimators that need them, and are installed through extras
(``pip install "reguq[boosting,torch]"``).
"""

from __future__ import annotations

from . import conformal, metrics, probabilistic, quantile, splits
from ._base import BaseIntervalRegressor, check_alpha
from ._version import __version__
from .splits import train_calib_split

__all__ = [
    "BaseIntervalRegressor",
    "__version__",
    "check_alpha",
    "conformal",
    "metrics",
    "probabilistic",
    "quantile",
    "splits",
    "train_calib_split",
]
