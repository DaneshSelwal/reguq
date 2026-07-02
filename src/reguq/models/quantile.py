"""Quantile Regressor."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseUQRegressor


class QuantileRegressor(BaseUQRegressor):
    """Object-Oriented wrapper for Quantile Regression.
    
    This regressor trains two distinct quantile models: one for the lower bound,
    and one for the upper bound. The base estimator MUST support quantile 
    regression (e.g., GradientBoostingRegressor(loss='quantile'), LightGBM, etc.).
    
    You should provide instantiated lower and upper estimators, or a single 
    estimator if you will pass `alpha` at predict time and rely on the wrapper
    to set the quantiles via `set_params(alpha=...)` (assuming the estimator
    supports an `alpha` or `quantile` parameter).
    """
    
    def __init__(self, lower_estimator: Any, upper_estimator: Any = None):
        """
        Args:
            lower_estimator: Estimator for the lower quantile. If `upper_estimator` 
                is None, this will be cloned and its parameter modified.
            upper_estimator: Estimator for the upper quantile.
        """
        self.lower_estimator = lower_estimator
        self.upper_estimator = upper_estimator if upper_estimator is not None else clone(lower_estimator)
        
        self.lower_estimator_ = None
        self.upper_estimator_ = None
        self.alpha_ = None
        
        # State
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> QuantileRegressor:
        """Fit the underlying models.
        
        Note: The actual fitting for the exact quantiles might happen at fit time
        if the alpha is known, or delayed to predict time. To keep the sklearn API,
        we usually fit here. But since `predict` takes `alpha`, we might need to 
        refit if `alpha` changes. 
        To optimize, we store the data and fit upon the first predict, or if a user 
        wants to fit immediately, we assume alpha=0.1.
        """
        if hasattr(X, "to_numpy"):
            self.X_train_ = X.to_numpy()
        else:
            self.X_train_ = np.asarray(X)
            
        if hasattr(y, "to_numpy"):
            self.y_train_ = y.to_numpy().ravel()
        else:
            self.y_train_ = np.asarray(y).ravel()
            
        return self
        
    def _set_quantile_param(self, estimator: Any, q: float):
        """Try to set the quantile parameter for known estimators."""
        # Check for LightGBM, XGBoost, sklearn GradientBoosting
        valid_params = ["alpha", "quantile", "q"]
        params = estimator.get_params()
        for p in valid_params:
            if p in params:
                estimator.set_params(**{p: q})
                return
        
        # If not standard, assume it's already configured correctly 
        # by the user before passing into this wrapper.

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the target and uncertainty intervals.
        
        Args:
            X: Input features.
            alpha: Miscoverage rate. Lower quantile = alpha/2, Upper = 1 - alpha/2.
        """
        if self.X_train_ is None:
            raise ValueError("This QuantileRegressor instance is not fitted yet. Call 'fit' first.")
            
        # Fit models if alpha changed or not fitted yet
        if self.alpha_ != alpha or self.lower_estimator_ is None:
            q_low = alpha / 2.0
            q_high = 1.0 - (alpha / 2.0)
            
            self.lower_estimator_ = clone(self.lower_estimator)
            self.upper_estimator_ = clone(self.upper_estimator)
            
            self._set_quantile_param(self.lower_estimator_, q_low)
            self._set_quantile_param(self.upper_estimator_, q_high)
            
            self.lower_estimator_.fit(self.X_train_, self.y_train_)
            self.upper_estimator_.fit(self.X_train_, self.y_train_)
            self.alpha_ = alpha
            
        if hasattr(X, "to_numpy"):
            X_test = X.to_numpy()
        else:
            X_test = np.asarray(X)
            
        y_low = np.asarray(self.lower_estimator_.predict(X_test)).ravel()
        y_high = np.asarray(self.upper_estimator_.predict(X_test)).ravel()
        
        # Ensure correct ordering
        y_lower = np.minimum(y_low, y_high)
        y_upper = np.maximum(y_low, y_high)
        y_pred = (y_lower + y_upper) / 2.0
        
        return y_pred, y_lower, y_upper
