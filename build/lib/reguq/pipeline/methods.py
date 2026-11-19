"""Method table: name -> builder.

``METHODS`` wrap a point regressor (one row per model x method);
``MODEL_FREE`` methods bring their own model (one row each). Register a new
method by adding an entry; builders receive ``(model, config, params)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .. import conformal as C
from .. import probabilistic as P
from .. import quantile as Q

if TYPE_CHECKING:
    from .config import PipelineConfig

Builder = Callable[[Any, "PipelineConfig", dict[str, Any]], Any]


def _cal(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "calib_size": cfg.calib_size,
        "time_ordered": cfg.time_ordered,
        "random_state": cfg.random_state,
    }


METHODS: dict[str, Builder] = {
    "split": lambda m, c, p: C.SplitConformalRegressor(m, **_cal(c), **p),
    "normalized": lambda m, c, p: C.SplitConformalRegressor(m, score="normalized", **_cal(c), **p),
    "cqr": lambda m, c, p: C.ConformalizedQuantileRegressor(m, alpha=c.alpha, **_cal(c), **p),
    "cv_plus": lambda m, c, p: C.CrossConformalRegressor(
        m, shuffle=not c.time_ordered, random_state=c.random_state, **p
    ),
    "jab": lambda m, c, p: C.JackknifePlusAfterBootstrapRegressor(
        m, random_state=c.random_state, **p
    ),
    "nexcp": lambda m, c, p: C.NexCPRegressor(m, calib_size=c.calib_size, **p),
    "aci": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="aci", alpha=c.alpha, method_params=p or None
    ),
    "dtaci": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="dtaci", alpha=c.alpha, method_params=p or None
    ),
    "sf_ogd": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="sf_ogd", alpha=c.alpha, method_params=p or None
    ),
    "saocp": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="saocp", alpha=c.alpha, method_params=p or None
    ),
    "nexcp_online": lambda m, c, p: C.OnlineConformalRegressor(
        m, method="nexcp", alpha=c.alpha, method_params=p or None
    ),
    "extreme": lambda m, c, p: C.ExtremeValueConformalRegressor(m, **_cal(c), **p),
    "acp": lambda m, c, p: C.AdaptiveCoveragePolicyRegressor(m, alpha=c.alpha, **_cal(c), **p),
    "multi_quantile": lambda m, c, p: Q.MultiQuantileRegressor(m, **p),
    "ibug": lambda m, c, p: P.IBUGRegressor(
        m, time_ordered=c.time_ordered, random_state=c.random_state, **p
    ),
    "card": lambda m, c, p: P.CARDRegressor(m, random_state=c.random_state, **p),
}

MODEL_FREE: dict[str, Callable[[PipelineConfig, dict[str, Any]], Any]] = {
    "ngboost": lambda c, p: P.NGBoostRegressor(random_state=c.random_state, **p),
    "pgbm": lambda c, p: P.PGBMRegressor(random_state=c.random_state, **p),
    "hcm": lambda c, p: P.HCMRegressor(
        time_ordered=c.time_ordered, random_state=c.random_state, **p
    ),
    "treeffuser": lambda c, p: P.TreeffuserRegressor(seed=c.random_state, **p),
    "qrf": lambda c, p: Q.QuantileForestRegressor(random_state=c.random_state, **p),
}
