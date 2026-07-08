"""Advanced probabilistic regression methods (CARD, Treeffuser, IBUG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random

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
        from sklearn.preprocessing import StandardScaler
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = _to_numpy(y)

        # Scale features
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(X_np)
        if self.feature_scaler.scale_ is not None:
            self.feature_scaler.scale_ = np.where(self.feature_scaler.scale_ == 0.0, 1.0, self.feature_scaler.scale_)
        X_scaled = self.feature_scaler.transform(X_np)

        # Get base predictions and residuals
        y_pred = self.base_model.predict(X_np)
        residuals = y_np - y_pred

        # Scale residuals
        self.residual_scaler = StandardScaler()
        self.residual_scaler.fit(residuals.reshape(-1, 1))
        if self.residual_scaler.scale_ is not None:
            self.residual_scaler.scale_ = np.where(self.residual_scaler.scale_ == 0.0, 1.0, self.residual_scaler.scale_)
        residuals_scaled = self.residual_scaler.transform(residuals.reshape(-1, 1)).ravel()

        # Setup linear beta schedule
        betas = np.linspace(1e-4, 0.02, self.T)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32).to(self.device)

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        r_t = torch.tensor(residuals_scaled, dtype=torch.float32).to(self.device)

        # Build MLP. Input dimension: X features + 1 (for r_t_noisy) + 1 (for t)
        self.mlp = nn.Sequential(
            nn.Linear(X_scaled.shape[1] + 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for _ in range(self.epochs):
            t = torch.randint(0, self.T, (len(X_t),), dtype=torch.long).to(self.device)
            noise = torch.randn_like(r_t)

            alpha_bar_t = self.alphas_cumprod[t]
            r_t_noisy = torch.sqrt(alpha_bar_t) * r_t + torch.sqrt(1.0 - alpha_bar_t) * noise

            t_normalized = t.float() / float(self.T)
            mlp_input = torch.cat([X_t, r_t_noisy.unsqueeze(1), t_normalized.unsqueeze(1)], dim=1)
            pred_noise = self.mlp(mlp_input).squeeze()

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

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        base_pred = self.base_model.predict(X_np)

        X_scaled = self.feature_scaler.transform(X_np)

        betas = torch.tensor(np.linspace(1e-4, 0.02, self.T), dtype=torch.float32).to(self.device)
        alphas = 1.0 - betas

        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        samples = []
        for _ in range(self.n_samples):
            r_curr = torch.randn(len(X_t), device=self.device)
            for t_idx in reversed(range(self.T)):
                t_tensor = torch.full((len(X_t),), t_idx, dtype=torch.long, device=self.device)
                t_normalized = t_tensor.float() / float(self.T)

                mlp_input = torch.cat([X_t, r_curr.unsqueeze(1), t_normalized.unsqueeze(1)], dim=1)
                with torch.no_grad():
                    pred_noise = self.mlp(mlp_input).squeeze()

                beta_t = betas[t_idx]
                alpha_t = alphas[t_idx]
                alpha_bar_t = self.alphas_cumprod[t_idx]

                mean = (r_curr - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise) / torch.sqrt(alpha_t)

                if t_idx > 0:
                    z = torch.randn_like(r_curr)
                    sigma_t = torch.sqrt(beta_t)
                    r_curr = mean + sigma_t * z
                else:
                    r_curr = mean

            r_curr_np = r_curr.cpu().numpy()
            r_unscaled = self.residual_scaler.inverse_transform(r_curr_np.reshape(-1, 1)).ravel()
            samples.append(base_pred + r_unscaled)

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
        candidate_k: Candidate k values for validation tuning.
    """

    def __init__(self, base_model, n_neighbors: int | None = None, candidate_k: list[int] | None = None):
        self.base_model = base_model
        self.n_neighbors = n_neighbors
        self.candidate_k = candidate_k or [10, 20, 50, 100, 200]
        self._train_residuals = None
        self._train_leaves = None
        self._fitted = False

    def _get_leaves(self, X_np, model=None):
        estimator = model if model is not None else self.base_model
        if hasattr(estimator, "predict"):
            try:
                # LightGBM
                if hasattr(estimator, "booster_"):
                    return estimator.booster_.predict(X_np, pred_leaf=True)
                # XGBoost
                elif hasattr(estimator, "get_booster"):
                    import xgboost as xgb
                    dmat = xgb.DMatrix(X_np)
                    return estimator.get_booster().predict(dmat, pred_leaf=True)
                # CatBoost
                elif hasattr(estimator, "calc_leaf_indexes"):
                    return estimator.calc_leaf_indexes(X_np)
            except Exception:
                pass
        return None

    def fit(self, X, y):
        """Fit IBUG by storing training data leaf assignments.

        Args:
            X: Training features.
            y: Training targets.
        """
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = _to_numpy(y)

        # Validation-set tuning if n_neighbors is not set
        if self.n_neighbors is None:
            from sklearn.model_selection import train_test_split
            from sklearn.base import clone
            X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

            cloned_model = clone(self.base_model)
            cloned_model.fit(X_tr, y_tr)

            tr_leaves = self._get_leaves(X_tr, model=cloned_model)
            val_leaves = self._get_leaves(X_val, model=cloned_model)

            tr_preds = cloned_model.predict(X_tr)
            tr_res = y_tr - tr_preds

            val_preds = cloned_model.predict(X_val)
            val_res = y_val - val_preds

            best_k = 50
            best_nll = float("inf")
            n_train = len(X_tr)

            candidates = [k for k in self.candidate_k if k <= n_train]
            if not candidates:
                candidates = [max(1, n_train - 1)]

            if tr_leaves is not None and val_leaves is not None:
                n_val = len(X_val)
                matches = np.zeros((n_val, n_train), dtype=np.int32)
                batch_size = 250
                for start_idx in range(0, n_val, batch_size):
                    end_idx = min(start_idx + batch_size, n_val)
                    val_batch = val_leaves[start_idx:end_idx]
                    matches[start_idx:end_idx] = np.sum(
                        val_batch[:, np.newaxis, :] == tr_leaves[np.newaxis, :, :], axis=2
                    )

                for k in candidates:
                    top_indices = np.argpartition(matches, -k, axis=1)[:, -k:]
                    val_stds = np.std(tr_res[top_indices], axis=1)
                    val_stds = _safe_sigma(val_stds)
                    nll = 0.5 * np.log(2 * np.pi * val_stds**2) + 0.5 * (val_res / val_stds)**2
                    mean_nll = np.mean(nll)
                    if mean_nll < best_nll:
                        best_nll = mean_nll
                        best_k = k
            else:
                from sklearn.neighbors import NearestNeighbors
                knn = NearestNeighbors(n_neighbors=min(max(candidates), len(X_tr)))
                knn.fit(X_tr)
                _, indices = knn.kneighbors(X_val)
                for k in candidates:
                    val_stds = np.std(tr_res[indices[:, :k]], axis=1)
                    val_stds = _safe_sigma(val_stds)
                    nll = 0.5 * np.log(2 * np.pi * val_stds**2) + 0.5 * (val_res / val_stds)**2
                    mean_nll = np.mean(nll)
                    if mean_nll < best_nll:
                        best_nll = mean_nll
                        best_k = k
            
            self.n_neighbors = best_k

        y_pred = self.base_model.predict(X_np)
        self._train_residuals = y_np - y_pred
        self._train_leaves = self._get_leaves(X_np)

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
        test_leaves = self._get_leaves(X_np)

        if self._train_leaves is not None and test_leaves is not None:
            try:
                k = min(self.n_neighbors, self._train_leaves.shape[0])
                if k < 1:
                    k = 1
                # Vectorized match comparison in batches of 250
                batch_size = 250
                for start_idx in range(0, len(X_np), batch_size):
                    end_idx = min(start_idx + batch_size, len(X_np))
                    test_batch = test_leaves[start_idx:end_idx]

                    matches = np.sum(test_batch[:, np.newaxis, :] == self._train_leaves[np.newaxis, :, :], axis=2)
                    top_indices = np.argpartition(matches, -k, axis=1)[:, -k:]
                    stds[start_idx:end_idx] = np.std(self._train_residuals[top_indices], axis=1)

            except Exception:
                stds = np.full(len(X_np), np.std(self._train_residuals))
        else:
            # Fallback: use KNN on features
            from sklearn.neighbors import NearestNeighbors

            knn = NearestNeighbors(n_neighbors=min(self.n_neighbors, len(self._X_train)))
            knn.fit(self._X_train)
            _, indices = knn.kneighbors(X_np)
            stds = np.std(self._train_residuals[indices], axis=1)

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
            from treeffuser.tree_score_model import TreeBasedScoreModel
        except ImportError:
            raise ImportError(
                "Treeffuser not installed. Install with: pip install treeffuser"
            )

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        y_np = _to_numpy(y)

        # Extract parameters and class name from base_model
        model_name = self.base_model.__class__.__name__.lower()
        if "lgbm" in model_name or "lightgbm" in model_name:
            model_name = "lightgbm"
        elif "xgb" in model_name:
            model_name = "xgboost"
        elif "catboost" in model_name:
            model_name = "catboost"
        elif "gpboost" in model_name:
            model_name = "gpboost"
        elif "gradientboosting" in model_name or "gradient_boosting" in model_name:
            model_name = "gradient_boosting"
        else:
            model_name = "lightgbm"

        model_params = self.base_model.get_params()
        score_model = TreeBasedScoreModel(model_name=model_name, model_params=model_params)
        self._treeffuser = Treeffuser(score_model=score_model)
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
# Hyperspherical Confidence Mapping (HCM)
# =============================================================================

class HCMUCIRegressor(nn.Module):
    """Paper-style tabular regression backbone: 3 hidden layers of width 20 with LeakyReLU."""

    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 2)   # direction d
        self.fc5 = nn.Linear(20, 1)   # magnitude R
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        d = self.fc4(x)
        R = self.fc5(x)
        return R, d


class HCMRegressor:
    """Hyperspherical Confidence Mapping (HCM) Regressor wrapper."""

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
        patience: int = 15,
        weight_decay: float = 1e-5,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device
        self.model = None
        self.scale = 1.0
        self._fitted = False

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        y_np = _to_numpy(y).astype(np.float32)

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        self.feature_scaler.fit(X_np)
        if self.feature_scaler.scale_ is not None:
            self.feature_scaler.scale_ = np.where(self.feature_scaler.scale_ == 0.0, 1.0, self.feature_scaler.scale_)
        X_scaled = self.feature_scaler.transform(X_np)

        self.target_scaler.fit(y_np.reshape(-1, 1))
        if self.target_scaler.scale_ is not None:
            self.target_scaler.scale_ = np.where(self.target_scaler.scale_ == 0.0, 1.0, self.target_scaler.scale_)
        y_scaled = self.target_scaler.transform(y_np.reshape(-1, 1)).ravel()

        X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=self.seed)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.model = HCMUCIRegressor(X_tr.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(np.concatenate([y_tr.reshape(-1, 1), y_tr.reshape(-1, 1)], axis=1), dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        best_state = None
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for x_batch, y_expanded in train_loader:
                x_batch = x_batch.to(self.device)
                y_expanded = y_expanded.to(self.device)

                optimizer.zero_grad()
                pred_R, pred_d = self.model(x_batch)

                R_target = torch.sqrt(torch.sum(y_expanded ** 2, dim=1, keepdim=True))
                d_target = y_expanded / (R_target + 1e-8)

                d_loss = criterion(pred_R * d_target, y_expanded)
                R_loss = criterion(R_target * pred_d, y_expanded)
                loss = d_loss + R_loss

                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                x_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                pred_R_val, pred_d_val = self.model(x_val_tensor)
                pred_y_val = (pred_R_val * pred_d_val)[:, 0].cpu().numpy()
                val_loss = float(np.mean((y_val - pred_y_val) ** 2))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Temperature Calibration on validation set
        self.model.eval()
        with torch.no_grad():
            x_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
            pred_R_val, pred_d_val = self.model(x_val_tensor)
            d_norm_sq_val = torch.sum(pred_d_val ** 2, dim=1)
            sigma_hat_val = torch.sqrt(torch.abs(d_norm_sq_val - 1.0)) * torch.abs(pred_R_val.squeeze(-1))
            raw_sigma_val = sigma_hat_val.cpu().numpy()
            pred_y_val = (pred_R_val * pred_d_val)[:, 0].cpu().numpy()

        absolute_errors_val = np.abs(y_val - pred_y_val)

        # Temperature grid search
        raw_sigma_val = np.clip(raw_sigma_val, 1e-8, None)
        grid = np.logspace(-2, 2, 400)
        target_coverage = np.array([0.68, 0.95, 0.997])
        best_scale = 1.0
        best_loss = np.inf
        for scale in grid:
            sigma_scaled = raw_sigma_val * scale
            coverage = np.array([
                np.mean(absolute_errors_val <= sigma_scaled),
                np.mean(absolute_errors_val <= 2.0 * sigma_scaled),
                np.mean(absolute_errors_val <= 3.0 * sigma_scaled)
            ])
            loss = np.sum((coverage - target_coverage) ** 2)
            if loss < best_loss:
                best_loss = loss
                best_scale = scale

        self.scale = float(best_scale)
        self._fitted = True
        return self

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        X_scaled = self.feature_scaler.transform(X_np)

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            pred_R, pred_d = self.model(x_tensor)
            d_norm_sq = torch.sum(pred_d ** 2, dim=1)
            sigma_hat = torch.sqrt(torch.abs(d_norm_sq - 1.0)) * torch.abs(pred_R.squeeze(-1))
            pred_y_scaled = (pred_R * pred_d)[:, 0].cpu().numpy()
            sigma_hat_scaled = sigma_hat.cpu().numpy()

        mean = self.target_scaler.inverse_transform(pred_y_scaled.reshape(-1, 1)).ravel()
        sigma = sigma_hat_scaled * self.scale * float(self.target_scaler.scale_[0])
        return mean, _safe_sigma(sigma)


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
    hcm_config: Mapping[str, Any] | None = None,
) -> PhaseResult:
    """Run advanced probabilistic regression methods.

    Supported methods:
    - card: CARD (Classification And Regression Diffusion)
    - ibug: IBUG (Instance-Based Uncertainty using Gradient Boosting)
    - treeffuser: Treeffuser diffusion models
    - hcm: Hyperspherical Confidence Mapping (HCM)

    Args:
        data: Input data (DataFrame, CSV path, or dict with train/test).
        target_col: Name of the target column.
        models: List of model IDs to use. Defaults to all supported models.
        params_source: Source for model parameters.
        output_config: Output configuration.
        split_config: Train/test split configuration.
        alpha: Significance level for intervals (default: 0.1).
        methods: List of methods to run (default: ["card", "ibug", "treeffuser", "hcm"]).
        card_config: Configuration for CARD (hidden_dim, T, epochs, n_samples).
        ibug_config: Configuration for IBUG (n_neighbors).
        hcm_config: Configuration for HCM (lr, epochs, batch_size, patience).

    Returns:
        PhaseResult with predictions and metrics.
    """
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_PROBABILISTIC)
    output_cfg = coerce_output_config(output_config)

    if not (0 < alpha < 1):
        raise ValueError("alpha must satisfy 0 < alpha < 1")

    methods = methods or ["card", "ibug", "treeffuser", "hcm"]
    card_cfg = dict(card_config or {})
    ibug_cfg = dict(ibug_config or {})
    hcm_cfg = dict(hcm_config or {})

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
                        n_neighbors=ibug_cfg.get("n_neighbors", None),
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

                elif method == "hcm":
                    wrapper = HCMRegressor(
                        lr=hcm_cfg.get("lr", 1e-3),
                        epochs=hcm_cfg.get("epochs", 200),
                        batch_size=hcm_cfg.get("batch_size", 64),
                        patience=hcm_cfg.get("patience", 15),
                        weight_decay=hcm_cfg.get("weight_decay", 1e-5),
                        seed=hcm_cfg.get("seed", 42),
                        device=hcm_cfg.get("device", "cpu"),
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
