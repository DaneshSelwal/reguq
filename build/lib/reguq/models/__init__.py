"""Model registry: build point regressors by name, importing backends lazily.

>>> from reguq.models import make_model
>>> make_model("randomforest", {"n_estimators": 50}, random_state=0)  # doctest: +ELLIPSIS
RandomForestRegressor(...)
"""

from __future__ import annotations

from .registry import available_models, make_model, register_model

__all__ = ["available_models", "make_model", "register_model"]
