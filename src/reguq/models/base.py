"""Base classes for Object-Oriented UQ Models."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class BaseUQRegressor(BaseEstimator, RegressorMixin):
    """Base class for all Uncertainty Quantification regressors."""
    
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> BaseUQRegressor:
        """Fit the model to data. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement fit()")
        
    def predict(self, X: np.ndarray | pd.DataFrame, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the target and uncertainty intervals.
        
        Args:
            X: Input features.
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage).
            
        Returns:
            Tuple of (y_pred, y_lower, y_upper).
        """
        raise NotImplementedError("Subclasses must implement predict()")
        
    def plot_predictions(
        self,
        X: np.ndarray | pd.DataFrame,
        y_true: np.ndarray | pd.Series | None = None,
        alpha: float = 0.1,
        backend: str = "matplotlib",
        max_points: int = 300,
        title: str | None = None,
    ) -> Any:
        """Plot the predictions and uncertainty intervals.
        
        Args:
            X: Input features.
            y_true: Optional true target values for comparison.
            alpha: Miscoverage rate for intervals.
            backend: Plotting backend ("matplotlib" or "plotly").
            max_points: Maximum number of points to plot (for readability).
            title: Custom title for the plot.
        """
        y_pred, y_lower, y_upper = self.predict(X, alpha=alpha)
        
        if len(y_pred) > max_points:
            idx = np.random.choice(len(y_pred), max_points, replace=False)
            idx.sort()
            y_pred = y_pred[idx]
            y_lower = y_lower[idx]
            y_upper = y_upper[idx]
            if y_true is not None:
                if isinstance(y_true, pd.Series):
                    y_true = y_true.iloc[idx].to_numpy()
                else:
                    y_true = np.asarray(y_true)[idx]
        else:
            if y_true is not None:
                if isinstance(y_true, pd.Series):
                    y_true = y_true.to_numpy()
                else:
                    y_true = np.asarray(y_true)

        title_str = title or f"{self.__class__.__name__} Predictions ({(1-alpha)*100:.0f}% Interval)"
        
        if backend.lower() == "plotly":
            return self._plot_plotly(y_pred, y_lower, y_upper, y_true, title_str)
        else:
            return self._plot_matplotlib(y_pred, y_lower, y_upper, y_true, title_str)
            
    def _plot_matplotlib(
        self,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        y_true: np.ndarray | None,
        title: str,
    ) -> Any:
        import matplotlib.pyplot as plt
        
        x = np.arange(len(y_pred))
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if y_true is not None:
            ax.plot(x, y_true, label="True Value", color="#1f77b4", linewidth=1.5)
            
        ax.plot(x, y_pred, label="Prediction", color="#ff7f0e", linewidth=1.5, linestyle="--")
        ax.fill_between(x, y_lower, y_upper, color="#2ca02c", alpha=0.3, label="Uncertainty Interval")
        
        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Target Value")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            plt.show()
            
        return fig
        
    def _plot_plotly(
        self,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        y_true: np.ndarray | None,
        title: str,
    ) -> Any:
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly to use backend='plotly'.")
            
        x = np.arange(len(y_pred))
        fig = go.Figure()
        
        if y_true is not None:
            fig.add_trace(go.Scatter(
                x=x, y=y_true,
                mode="lines",
                name="True Value",
                line=dict(color="#1f77b4", width=2)
            ))
            
        fig.add_trace(go.Scatter(
            x=x, y=y_pred,
            mode="lines",
            name="Prediction",
            line=dict(color="#ff7f0e", width=2, dash="dash")
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=y_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=y_lower,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(44, 160, 44, 0.3)",
            line=dict(width=0),
            name="Uncertainty Interval"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Sample Index",
            yaxis_title="Target Value",
            template="plotly_white",
            hovermode="x unified"
        )
        
        try:
            from IPython.display import display
            display(fig)
        except ImportError:
            fig.show()
            
        return fig
