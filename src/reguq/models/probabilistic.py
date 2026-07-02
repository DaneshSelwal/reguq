"""Probabilistic Regression Models."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone

from .base import BaseUQRegressor
from ..probabilistic_advanced import CARDRegressor, IBUGRegressor, TreeffuserWrapper


class ProbabilisticRegressor(BaseUQRegressor):
    """Object-Oriented wrapper for Probabilistic Regression.
    
    Supports:
        - 'card': Classification And Regression Diffusion
        - 'ibug': Instance-Based Uncertainty using Gradient Boosting
        - 'treeffuser': Treeffuser diffusion models
    """
    
    def __init__(
        self,
        base_estimator: Any,
        method: str = "ibug",
        # CARD / Treeffuser specific
        hidden_dim: int = 128,
        T: int = 50,
        epochs: int = 200,
        n_samples: int = 100,
        device: str = "cpu",
        # IBUG specific
        n_neighbors: int = 50,
    ):
        self.base_estimator = base_estimator
        self.method = method.lower()
        
        self.hidden_dim = hidden_dim
        self.T = T
        self.epochs = epochs
        self.n_samples = n_samples
        self.device = device
        self.n_neighbors = n_neighbors
        
        self.wrapper_ = None
        
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ProbabilisticRegressor:
        """Fit the probabilistic model."""
        base = clone(self.base_estimator)
        
        if self.method == "card":
            # CARD expects pre-fitted base model according to its init, wait, 
            # the original implementation does NOT fit the base_model inside fit().
            # Wait, looking at the code: base_model is used to predict, but it is not fitted inside CARDRegressor!
            # So we MUST fit it first.
            base.fit(X, y)
            self.wrapper_ = CARDRegressor(
                base, self.hidden_dim, self.T, 1e-3, self.epochs, self.n_samples, self.device
            )
        elif self.method == "ibug":
            base.fit(X, y)
            self.wrapper_ = IBUGRegressor(base, self.n_neighbors)
        elif self.method == "treeffuser":
            base.fit(X, y) # Treeffuser wrapper doesn't use base_model for predictions but stores it.
            self.wrapper_ = TreeffuserWrapper(base, self.n_samples)
        else:
            raise ValueError(f"Unknown probabilistic method: {self.method}")
            
        self.wrapper_.fit(X, y)
        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the target and uncertainty intervals.
        
        Converts mean and standard deviation to prediction intervals assuming
        a Gaussian distribution.
        """
        if self.wrapper_ is None:
            raise ValueError("This ProbabilisticRegressor is not fitted yet. Call 'fit' first.")
            
        mean, sigma = self.wrapper_.predict(X)
        
        z_low = norm.ppf(alpha / 2.0)
        z_high = norm.ppf(1.0 - alpha / 2.0)
        
        y_lower = mean + z_low * sigma
        y_upper = mean + z_high * sigma
        
        return mean, y_lower, y_upper
