"""Conformal Prediction Regressors."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import clone

from .base import BaseUQRegressor
from ..conformal_standard import _predict_mapie, _predict_puncc, _manual_split_conformal
from ..conformal_advanced import (
    _predict_nexcp_split,
    _predict_nexcp_full,
    _predict_nexcp_jackknife_ab,
    _predict_nexcp_cv_plus,
    _predict_online_split,
    _predict_faci,
    _predict_mfcs_split,
    _predict_mfcs_full,
    _predict_puncc_cvplus,
    _predict_puncc_cqr,
)


class ConformalRegressor(BaseUQRegressor):
    """Object-Oriented wrapper for Conformal Prediction methods.
    
    Supported methods:
        - 'mapie': Uses MapieRegressor (default)
        - 'puncc': Uses PUNCC SplitCP
        - 'manual_split': Fallback manual split conformal
        - 'nexcp_split': NexCP Split with exponential weighting
        - 'nexcp_full': NexCP Full conformal
        - 'nexcp_jackknife_ab': NexCP Jackknife+ after Bootstrap
        - 'nexcp_cv_plus': NexCP Cross-Validation Plus
        - 'online_split': Online Split conformal
        - 'faci': Fully Adaptive Conformal Inference
        - 'mfcs_split': Model-Free Conformal Selection (Split)
        - 'mfcs_full': Model-Free Conformal Selection (Full)
        - 'cvplus': PUNCC CV+ (Cross-Validation Plus)
        - 'cqr': PUNCC CQR (Conformalized Quantile Regression)
    """
    
    def __init__(
        self,
        base_estimator: Any,
        method: str = "mapie",
        calibration_size: float = 0.2,
        random_state: int = 42,
        # Method-specific kwargs
        mapie_method: str = "plus",
        decay: float = 0.99,
        n_folds: int = 5,
        n_bootstrap: int = 50,
        gamma: float = 0.01,
    ):
        self.base_estimator = base_estimator
        self.method = method
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.mapie_method = mapie_method
        self.decay = decay
        self.n_folds = n_folds
        self.n_bootstrap = n_bootstrap
        self.gamma = gamma
        
        self.estimator_ = None
        
        # State tracking for some methods
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ConformalRegressor:
        """Fit the underlying model and prepare for conformal prediction.
        
        For many conformal methods, the calibration is done inside predict() 
        or it requires X_train/y_train during prediction. We store them.
        """
        self.estimator_ = clone(self.base_estimator)
        
        if hasattr(X, "to_numpy"):
            self.X_train_ = X.to_numpy()
        else:
            self.X_train_ = np.asarray(X)
            
        if hasattr(y, "to_numpy"):
            self.y_train_ = y.to_numpy().ravel()
        else:
            self.y_train_ = np.asarray(y).ravel()
            
        # We pre-fit the estimator on the entire dataset for some methods,
        # but the specific conformal backend might refit internally.
        # We will just rely on the backends inside predict().
        
        return self

    def predict(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float = 0.1,
        y_true: np.ndarray | pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the target and uncertainty intervals.
        
        Args:
            X: Input features to predict on.
            alpha: Miscoverage rate.
            y_true: Target values for test data, required only for FACI.
        """
        if self.estimator_ is None or self.X_train_ is None:
            raise ValueError("This ConformalRegressor instance is not fitted yet. Call 'fit' with appropriate arguments.")
            
        if hasattr(X, "to_numpy"):
            X_test = X.to_numpy()
        else:
            X_test = np.asarray(X)

        if y_true is not None:
            if hasattr(y_true, "to_numpy"):
                y_test = y_true.to_numpy().ravel()
            else:
                y_test = np.asarray(y_true).ravel()
        else:
            y_test = None

        m = self.method.lower()
        
        try:
            if m == "mapie":
                y_pred, y_lower, y_upper = _predict_mapie(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha, self.mapie_method
                )
            elif m == "puncc":
                y_pred, y_lower, y_upper = _predict_puncc(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha
                )
            elif m == "manual_split":
                y_pred, y_lower, y_upper = _manual_split_conformal(
                    clone(self.base_estimator), pd.DataFrame(self.X_train_), pd.Series(self.y_train_), pd.DataFrame(X_test), 
                    alpha, self.calibration_size, self.random_state
                )
            elif m == "nexcp_split":
                y_pred, y_lower, y_upper = _predict_nexcp_split(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha, self.decay
                )
            elif m == "nexcp_full":
                y_pred, y_lower, y_upper = _predict_nexcp_full(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha
                )
            elif m == "nexcp_jackknife_ab":
                def model_builder(): return clone(self.base_estimator)
                y_pred, y_lower, y_upper = _predict_nexcp_jackknife_ab(
                    model_builder, self.X_train_, self.y_train_, X_test, alpha, self.n_bootstrap, self.random_state
                )
            elif m == "nexcp_cv_plus":
                def model_builder(): return clone(self.base_estimator)
                y_pred, y_lower, y_upper = _predict_nexcp_cv_plus(
                    model_builder, self.X_train_, self.y_train_, X_test, alpha, self.n_folds, self.random_state
                )
            elif m == "online_split":
                y_pred, y_lower, y_upper = _predict_online_split(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha
                )
            elif m == "faci":
                if y_test is None:
                    raise ValueError("Method 'faci' requires true labels (y_true) passed to predict() to adapt coverage.")
                y_pred, y_lower, y_upper = _predict_faci(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, y_test, alpha, self.gamma
                )
            elif m == "mfcs_split":
                y_pred, y_lower, y_upper = _predict_mfcs_split(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha, self.calibration_size, self.random_state
                )
            elif m == "mfcs_full":
                y_pred, y_lower, y_upper = _predict_mfcs_full(
                    clone(self.base_estimator), self.X_train_, self.y_train_, X_test, alpha
                )
            elif m == "cvplus":
                y_pred, y_lower, y_upper = _predict_puncc_cvplus(
                    clone(self.base_estimator), pd.DataFrame(self.X_train_), pd.Series(self.y_train_), pd.DataFrame(X_test), alpha
                )
            elif m == "cqr":
                # Provide dummy lower/upper identical estimators since CQR transforms them internally
                y_pred, y_lower, y_upper = _predict_puncc_cqr(
                    clone(self.base_estimator), clone(self.base_estimator), pd.DataFrame(self.X_train_), pd.Series(self.y_train_), pd.DataFrame(X_test), alpha
                )
            else:
                raise ValueError(f"Unknown conformal method: {m}")
        except Exception as e:
            # Fallback
            import warnings
            warnings.warn(f"Method '{m}' failed with error: {e}. Falling back to 'manual_split'.")
            y_pred, y_lower, y_upper = _manual_split_conformal(
                clone(self.base_estimator), pd.DataFrame(self.X_train_), pd.Series(self.y_train_), pd.DataFrame(X_test), 
                alpha, self.calibration_size, self.random_state
            )

        return np.asarray(y_pred).ravel(), np.asarray(y_lower).ravel(), np.asarray(y_upper).ravel()
        
    def plot_predictions(
        self,
        X: np.ndarray | pd.DataFrame,
        y_true: np.ndarray | pd.Series | None = None,
        alpha: float = 0.1,
        backend: str = "matplotlib",
        max_points: int = 300,
        title: str | None = None,
    ):
        # We need to override this just in case method is FACI which requires y_true in predict()
        if self.method.lower() == "faci" and y_true is not None:
            y_pred, y_lower, y_upper = self.predict(X, alpha=alpha, y_true=y_true)
            
            if len(y_pred) > max_points:
                idx = np.random.choice(len(y_pred), max_points, replace=False)
                idx.sort()
                y_pred = y_pred[idx]
                y_lower = y_lower[idx]
                y_upper = y_upper[idx]
                if isinstance(y_true, pd.Series):
                    y_t = y_true.iloc[idx].to_numpy()
                else:
                    y_t = np.asarray(y_true)[idx]
            else:
                y_t = np.asarray(y_true)
                
            title_str = title or f"{self.__class__.__name__} ({self.method}) Predictions"
            if backend.lower() == "plotly":
                return self._plot_plotly(y_pred, y_lower, y_upper, y_t, title_str)
            else:
                return self._plot_matplotlib(y_pred, y_lower, y_upper, y_t, title_str)
        else:
            return super().plot_predictions(X, y_true, alpha, backend, max_points, title)
