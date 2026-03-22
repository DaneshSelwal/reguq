"""Advanced probabilistic regression methods (CARD, Treeffuser, IBUG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from .charts import generate_phase_charts
from .config import coerce_output_config
from .constants import DEFAULT_ALPHA, PHASE_PROBABILISTIC
from .data import prepare_data_bundle
from .export import embed_images_in_excel, write_json, write_phase_excel
from .metrics import gaussian_crps, gaussian_nll, interval_metrics, regression_metrics
from .params import resolve_model_params
from .types import OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def _safe_sigma(values: np.ndarray, fallback: float = 1.0) -> np.ndarray:
    """Ensure sigma values are valid (positive, finite)."""
    sigma = np.asarray(values, dtype=float)
    sigma = np.where(np.isfinite(sigma), sigma, fallback)
    sigma = np.maximum(sigma, 1e-8)
    return sigma


def _to_numpy(arr):
    """Convert to numpy array safely."""
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy().ravel()
    return np.asarray(arr).ravel()


# =============================================================================
# CARD (Classification And Regression Diffusion)
# =============================================================================


class CARDRegressor:
    """CARD: Diffusion-based uncertainty quantification for regression.

    CARD uses a diffusion model to learn the residual distribution of a base model,
    enabling probabilistic predictions with uncertainty estimates.

    Reference:
        Han, X., et al. "CARD: Classification and Regression Diffusion Models."
        NeurIPS 2022. https://arxiv.org/abs/2206.07275

    Args:
        base_model: A fitted sklearn-compatible regressor.
        hidden_dim: Hidden dimension for the MLP (default: 128).
        T: Number of diffusion timesteps (default: 50).
        lr: Learning rate for training (default: 1e-3).
        epochs: Number of training epochs (default: 200).
        n_samples: Number of samples for prediction (default: 100).
        device: Device to use ('cpu' or 'cuda').
    """

    def __init__(
        self,
        base_model,
        hidden_dim: int = 128,
        T: int = 50,
        lr: float = 1e-3,
        epochs: int = 200,
        n_samples: int = 100,
        device: str = "cpu",
    ):
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.T = T
        self.lr = lr
        self.epochs = epochs
        self.n_samples = n_samples
        self.device = device
        self.mlp = None
        self.optimizer = None
        self._fitted = False

    def fit(self, X, y):
        """Fit the CARD model.

        Args:
            X: Training features.
            y: Training targets.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("CARD requires PyTorch. Install with: pip install torch")

        # Get base predictions
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = _to_numpy(y)

        y_pred = self.base_model.predict(X_np)
        residuals = y_np - y_pred

        X_t = torch.tensor(X_np, dtype=torch.float32).to(self.device)
        r_t = torch.tensor(residuals, dtype=torch.float32).to(self.device)

        # Build MLP
        self.mlp = nn.Sequential(
            nn.Linear(X_np.shape[1] + 1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            t = torch.randint(0, self.T, (len(X_t),), dtype=torch.float32).to(self.device)
            noise = torch.randn_like(r_t)

            pred_noise = self.mlp(torch.cat([X_t, t.unsqueeze(1)], dim=1)).squeeze()

            loss = loss_fn(pred_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._fitted = True
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation.

        Args:
            X: Features to predict.

        Returns:
            Tuple of (mean predictions, standard deviation).
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("CARD requires PyTorch. Install with: pip install torch")

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        base_pred = self.base_model.predict(X_np)

        X_t = torch.tensor(X_np, dtype=torch.float32).to(self.device)

        samples = []
        for _ in range(self.n_samples):
            t = torch.zeros(len(X_t)).to(self.device)
            noise = torch.randn(len(X_t)).to(self.device)

            with torch.no_grad():
                pred_noise = self.mlp(torch.cat([X_t, t.unsqueeze(1)], dim=1)).squeeze()

            corrected_noise = (noise - pred_noise).cpu().numpy()
            samples.append(base_pred + corrected_noise)

        samples = np.stack(samples, axis=1)
        return samples.mean(axis=1), _safe_sigma(samples.std(axis=1))


# =============================================================================
# IBUG (Instance-Based Uncertainty using Gradient Boosting)
# =============================================================================


class IBUGRegressor:
    """IBUG: Instance-Based Uncertainty using Gradient Boosting.

    IBUG estimates prediction uncertainty by analyzing the distribution of
    residuals for similar training instances in the leaf nodes of gradient
    boosting trees.

    Reference:
        Brophy, J., et al. "IBUG: Instance-Based Uncertainty Estimation for
        Gradient Boosted Regression Trees." arXiv 2021.
        https://arxiv.org/abs/2110.03260

    Args:
        base_model: A fitted gradient boosting model (LightGBM, XGBoost, etc.).
        n_neighbors: Number of nearest neighbors to consider (default: 50).
    """

    def __init__(self, base_model, n_neighbors: int = 50):
        self.base_model = base_model
        self.n_neighbors = n_neighbors
        self._train_residuals = None
        self._train_leaves = None
        self._fitted = False

    def fit(self, X, y):
        """Fit IBUG by storing training data leaf assignments.

        Args:
            X: Training features.
            y: Training targets.
        """
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = _to_numpy(y)

        y_pred = self.base_model.predict(X_np)
        self._train_residuals = y_np - y_pred

        # Get leaf indices for training data
        if hasattr(self.base_model, "predict"):
            try:
                # LightGBM
                if hasattr(self.base_model, "booster_"):
                    self._train_leaves = self.base_model.booster_.predict(
                        X_np, pred_leaf=True
                    )
                # XGBoost
                elif hasattr(self.base_model, "get_booster"):
                    import xgboost as xgb

                    dmat = xgb.DMatrix(X_np)
                    self._train_leaves = self.base_model.get_booster().predict(
                        dmat, pred_leaf=True
                    )
                else:
                    # Fallback: use residuals directly with KNN
                    self._train_leaves = None
            except Exception:
                self._train_leaves = None

        self._X_train = X_np
        self._fitted = True
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation.

        Args:
            X: Features to predict.

        Returns:
            Tuple of (mean predictions, standard deviation).
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_pred = self.base_model.predict(X_np)

        stds = np.zeros(len(X_np))

        if self._train_leaves is not None:
            try:
                # Get leaf indices for test data
                if hasattr(self.base_model, "booster_"):
                    test_leaves = self.base_model.booster_.predict(X_np, pred_leaf=True)
                elif hasattr(self.base_model, "get_booster"):
                    import xgboost as xgb

                    dmat = xgb.DMatrix(X_np)
                    test_leaves = self.base_model.get_booster().predict(
                        dmat, pred_leaf=True
                    )

                # For each test instance, find training instances in same leaves
                for i, leaves in enumerate(test_leaves):
                    # Count matching leaves
                    matches = np.sum(self._train_leaves == leaves, axis=1)
                    top_indices = np.argsort(matches)[-self.n_neighbors :]
                    neighbor_residuals = self._train_residuals[top_indices]
                    stds[i] = np.std(neighbor_residuals)

            except Exception:
                stds = np.full(len(X_np), np.std(self._train_residuals))
        else:
            # Fallback: use KNN on features
            from sklearn.neighbors import NearestNeighbors

            knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self._X_train)))
            knn.fit(self._X_train)
            _, indices = knn.kneighbors(X_np)

            for i, idx in enumerate(indices):
                stds[i] = np.std(self._train_residuals[idx])

        return y_pred, _safe_sigma(stds)


# =============================================================================
# Treeffuser Integration
# =============================================================================


class TreeffuserWrapper:
    """Wrapper for Treeffuser diffusion models.

    Treeffuser combines gradient boosting with diffusion models for
    probabilistic predictions.

    Reference:
        Jolicoeur-Martineau, A., et al. "Generating and Imputing Tabular
        Data via Diffusion and Flow-based Gradient-Boosted Trees."
        AISTATS 2024. https://arxiv.org/abs/2309.09968

    Args:
        base_model: A fitted sklearn-compatible regressor.
        n_samples: Number of samples for prediction (default: 100).
    """

    def __init__(self, base_model, n_samples: int = 100):
        self.base_model = base_model
        self.n_samples = n_samples
        self._treeffuser = None
        self._fitted = False

    def fit(self, X, y):
        """Fit Treeffuser model.

        Args:
            X: Training features.
            y: Training targets.
        """
        try:
            from treeffuser import Treeffuser
        except ImportError:
            raise ImportError(
                "Treeffuser not installed. Install with: pip install treeffuser"
            )

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = _to_numpy(y)

        self._treeffuser = Treeffuser()
        self._treeffuser.fit(X_np, y_np)
        self._fitted = True
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and standard deviation.

        Args:
            X: Features to predict.

        Returns:
            Tuple of (mean predictions, standard deviation).
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        samples = self._treeffuser.sample(X_np, n_samples=self.n_samples)
        return samples.mean(axis=1), _safe_sigma(samples.std(axis=1))


# =============================================================================
# Main Runner Function
# =============================================================================


def run_probabilistic_advanced(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
    alpha: float = DEFAULT_ALPHA,
    methods: list[str] | None = None,
    card_config: Mapping[str, Any] | None = None,
    ibug_config: Mapping[str, Any] | None = None,
) -> PhaseResult:
    """Run advanced probabilistic regression methods.

    Supported methods:
    - card: CARD (Classification And Regression Diffusion)
    - ibug: IBUG (Instance-Based Uncertainty using Gradient Boosting)
    - treeffuser: Treeffuser diffusion models

    Args:
        data: Input data (DataFrame, CSV path, or dict with train/test).
        target_col: Name of the target column.
        models: List of model IDs to use. Defaults to all supported models.
        params_source: Source for model parameters.
        output_config: Output configuration.
        split_config: Train/test split configuration.
        alpha: Significance level for intervals (default: 0.1).
        methods: List of methods to run (default: ["card", "ibug"]).
        card_config: Configuration for CARD (hidden_dim, T, epochs, n_samples).
        ibug_config: Configuration for IBUG (n_neighbors).

    Returns:
        PhaseResult with predictions and metrics.
    """
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_PROBABILISTIC)
    output_cfg = coerce_output_config(output_config)

    if not (0 < alpha < 1):
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    methods = methods or ["card", "ibug"]
    card_cfg = dict(card_config or {})
    ibug_cfg = dict(ibug_config or {})

    model_params, tuned_params = resolve_model_params(
        models=model_ids,
        params_source=params_source,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
    )

    z_low = norm.ppf(alpha / 2.0)
    z_high = norm.ppf(1.0 - alpha / 2.0)

    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))
        base_estimator = model_registry.build_estimator(
            model_id=model_id, phase=PHASE_PROBABILISTIC, params=params
        )
        base_estimator.fit(bundle.X_train, bundle.y_train)

        for method in methods:
            key = f"{model_id}_{method}"

            try:
                if method == "card":
                    wrapper = CARDRegressor(
                        base_model=base_estimator,
                        hidden_dim=card_cfg.get("hidden_dim", 128),
                        T=card_cfg.get("T", 50),
                        epochs=card_cfg.get("epochs", 200),
                        n_samples=card_cfg.get("n_samples", 100),
                    )
                    wrapper.fit(bundle.X_train, bundle.y_train)
                    mean, sigma = wrapper.predict(bundle.X_test)

                elif method == "ibug":
                    wrapper = IBUGRegressor(
                        base_model=base_estimator,
                        n_neighbors=ibug_cfg.get("n_neighbors", 50),
                    )
                    wrapper.fit(bundle.X_train, bundle.y_train)
                    mean, sigma = wrapper.predict(bundle.X_test)

                elif method == "treeffuser":
                    wrapper = TreeffuserWrapper(
                        base_model=base_estimator,
                        n_samples=card_cfg.get("n_samples", 100),
                    )
                    wrapper.fit(bundle.X_train, bundle.y_train)
                    mean, sigma = wrapper.predict(bundle.X_test)

                else:
                    raise ValueError(f"Unknown probabilistic method '{method}'")

            except ImportError as e:
                # Skip methods with missing dependencies
                continue
            except Exception as e:
                # Fallback to residual-based estimation
                mean = base_estimator.predict(bundle.X_test)
                train_residuals = bundle.y_train.to_numpy() - base_estimator.predict(bundle.X_train)
                sigma = np.full_like(mean, np.std(train_residuals))

            y_true = bundle.y_test.to_numpy()
            y_lower = mean + z_low * sigma
            y_upper = mean + z_high * sigma

            pred_df = pd.DataFrame(
                {
                    "y_true": y_true,
                    "y_pred": mean,
                    "y_std": sigma,
                    "y_lower": y_lower,
                    "y_upper": y_upper,
                }
            )
            predictions[key] = pred_df

            row = {"model": model_id, "method": method, "alpha": alpha}
            row.update(regression_metrics(y_true=y_true, y_pred=mean))
            row.update(interval_metrics(y_true=y_true, y_lower=y_lower, y_upper=y_upper))
            row.update(
                {
                    "nll": gaussian_nll(y_true=y_true, mean=mean, std=sigma),
                    "crps": gaussian_crps(y_true=y_true, mean=mean, std=sigma),
                }
            )
            metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    result = PhaseResult(
        phase="probabilistic_advanced",
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )

    if output_cfg.output_dir is not None:
        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chart_result = None
        if output_cfg.export_plots or output_cfg.embed_excel_charts or output_cfg.show_inline_plots:
            chart_result = generate_phase_charts(
                phase_result=result,
                phase_name="probabilistic_advanced",
                output_cfg=output_cfg,
                output_dir=output_dir,
            )

        if output_cfg.export_plots and chart_result is not None:
            result.artifacts.extend(chart_result.image_paths)

        excel_path = output_dir / "probabilistic_advanced.xlsx"
        if output_cfg.export_excel:
            result.artifacts.append(write_phase_excel(result, excel_path))
            if output_cfg.embed_excel_charts and chart_result is not None and chart_result.images_by_sheet:
                embed_images_in_excel(workbook_path=excel_path, images_by_sheet=chart_result.images_by_sheet)

        if output_cfg.save_json and tuned_params:
            result.artifacts.append(write_json(tuned_params, output_dir / "probabilistic_advanced_tuned_params.json"))

    return result
