"""Advanced conformal prediction methods (NexCP, Adaptive CP, MFCS, CVPlus, CQR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import logging
import random
import numpy as np
import pandas as pd
from scipy.stats import genpareto
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split

from .charts import generate_conformal_charts
from .config import coerce_output_config
from .constants import DEFAULT_ALPHA, PHASE_CONFORMAL_ADVANCED
from .data import prepare_data_bundle
from .export import embed_images_in_excel, write_conformal_excel, write_json
from .metrics import interval_metrics, regression_metrics
from .params import resolve_model_params
from .types import ConformalResult, OutputConfig, PhaseResult, SplitConfig
import reguq.registry as model_registry


def _to_numpy(arr):
    """Convert to numpy array safely."""
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy().ravel()
    return np.asarray(arr).ravel()


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Compute weighted quantile."""
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(q, cdf, values))


def _extract_interval_bounds(intervals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower and upper bounds from interval arrays."""
    arr = np.asarray(intervals)
    if arr.ndim == 3:
        lower = arr[:, 0, 0]
        upper = arr[:, 1, 0]
        return np.asarray(lower).ravel(), np.asarray(upper).ravel()
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.asarray(arr[:, 0]).ravel(), np.asarray(arr[:, 1]).ravel()
    raise ValueError(f"Unsupported interval shape: {arr.shape}")


# =============================================================================
# PUNCC Advanced Methods (CVPlus, CQR)
# =============================================================================


def _predict_puncc_cvplus(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
    K: int = 5,
    random_state: int = 42,
):
    """PUNCC CV+ (Cross-Validation Plus) conformal prediction."""
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import CVPlus

    predictor = BasePredictor(estimator)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    cp = CVPlus(predictor, kfold=kf)
    cp.fit(X_train.to_numpy(), y_train.to_numpy())
    outputs = cp.predict(X_test.to_numpy(), alpha=alpha)

    if isinstance(outputs, tuple) and len(outputs) >= 2:
        y_pred = np.asarray(outputs[0]).ravel()
        intervals = outputs[1]
    else:
        raise ValueError("Unexpected PUNCC CVPlus predict output format.")

    y_lower, y_upper = _extract_interval_bounds(intervals)
    return y_pred, y_lower, y_upper


def _predict_puncc_cqr(
    lower_estimator,
    upper_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
    fit_ratio: float = 0.8,
    random_state: int = 42,
):
    """PUNCC CQR (Conformalized Quantile Regression) conformal prediction."""
    from deel.puncc.api.prediction import BasePredictor
    from deel.puncc.regression import CQR

    lower_predictor = BasePredictor(lower_estimator)
    upper_predictor = BasePredictor(upper_estimator)
    cp = CQR(lower_predictor, upper_predictor)

    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_np, y_train_np, train_size=fit_ratio, random_state=random_state
    )

    cp.fit(X_fit=X_fit, y_fit=y_fit, X_calib=X_cal, y_calib=y_cal)
    outputs = cp.predict(X_test.to_numpy(), alpha=alpha)

    if isinstance(outputs, tuple) and len(outputs) >= 2:
        y_pred = np.asarray(outputs[0]).ravel()
        intervals = outputs[1]
    else:
        raise ValueError("Unexpected PUNCC CQR predict output format.")

    y_lower, y_upper = _extract_interval_bounds(intervals)
    return y_pred, y_lower, y_upper


# =============================================================================
# NexCP Methods (Non-Exchangeable Conformal Prediction)
# =============================================================================


def _predict_nexcp_split(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    decay: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP Split with exponential weighting for non-exchangeable data."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    residuals = np.abs(y_train - yhat_train)
    weights = decay ** np.arange(len(residuals) - 1, -1, -1)
    q = _weighted_quantile(residuals, weights, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_nexcp_full(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP Full conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    residuals = np.abs(y_train - yhat_train)
    q = np.quantile(residuals, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_nexcp_jackknife_ab(
    model_builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_bootstrap: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP Jackknife+ after Bootstrap."""
    rng = np.random.default_rng(random_state)
    lowers, uppers, preds = [], [], []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        model = model_builder()
        model.fit(X_train[idx], y_train[idx])

        q = np.quantile(np.abs(y_train[idx] - model.predict(X_train[idx])), 1 - alpha)
        p = model.predict(X_test)

        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)

    y_pred = np.mean(preds, axis=0)
    y_lower = np.min(lowers, axis=0)
    y_upper = np.max(uppers, axis=0)
    return y_pred, y_lower, y_upper


def _predict_nexcp_cv_plus(
    model_builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NexCP CV+ (Cross-Validation Plus)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    lowers, uppers, preds = [], [], []

    for tr_idx, cal_idx in kf.split(X_train):
        model = model_builder()
        model.fit(X_train[tr_idx], y_train[tr_idx])

        q = np.quantile(np.abs(y_train[cal_idx] - model.predict(X_train[cal_idx])), 1 - alpha)
        p = model.predict(X_test)

        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)

    y_pred = np.mean(preds, axis=0)
    y_lower = np.mean(lowers, axis=0)
    y_upper = np.mean(uppers, axis=0)
    return y_pred, y_lower, y_upper


# =============================================================================
# Online/Adaptive CP Methods
# =============================================================================


def _predict_online_split(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online Split conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    residuals = np.abs(y_train - yhat_train)
    q = np.quantile(residuals, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_faci(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    window_grid: tuple[int, ...] = (25, 50, 100, 200, 400),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully Adaptive Conformal Inference (FACI) with window-grid pinball loss optimizer."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train
    for t in range(n_train, T):
        y_hat_all[t] = estimator.predict(X_all[t : t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)

    windows = list(window_grid)
    losses = np.zeros(len(windows))
    lower, upper = [], []
    for t in range(n_train, T):
        qs = []
        for i, w in enumerate(windows):
            start = max(0, t - w)
            q = np.quantile(residuals[start:t], 1 - alpha)
            qs.append(q)
        q_star = qs[np.argmin(losses)]
        lower.append(yhat_test[t - n_train] - q_star)
        upper.append(yhat_test[t - n_train] + q_star)

        r_t = residuals[t]
        for i, w in enumerate(windows):
            q = qs[i]
            losses[i] += alpha * q + (r_t - q) * (r_t > q)

    return yhat_test, np.array(lower), np.array(upper)


def _predict_saocp(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    window_min: int = 50,
    window_max: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Semi-Adaptive Online Conformal Prediction (SAOCP)."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train
    for t in range(n_train, T):
        y_hat_all[t] = estimator.predict(X_all[t : t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)
    lower, upper = [], []
    for t in range(n_train, T):
        w = min(max(window_min, int(t / 2)), window_max)
        start = max(0, t - w)
        q = np.quantile(residuals[start:t], 1 - alpha)
        lower.append(yhat_test[t - n_train] - q)
        upper.append(yhat_test[t - n_train] + q)

    return yhat_test, np.array(lower), np.array(upper)


def _predict_sf_ogd(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-Free Online Gradient Descent (SF-OGD)."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train
    for t in range(n_train, T):
        y_hat_all[t] = estimator.predict(X_all[t : t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)
    q_t = np.quantile(residuals[:n_train], 1 - alpha)
    lower, upper = [], []
    for t in range(n_train, T):
        lower.append(yhat_test[t - n_train] - q_t)
        upper.append(yhat_test[t - n_train] + q_t)

        r_t = residuals[t]
        grad = alpha - (r_t > q_t)
        eta = 1.0 / np.sqrt(t - n_train + 1)
        q_t = max(0.0, q_t - eta * grad)

    return yhat_test, np.array(lower), np.array(upper)


def _predict_online_cv_plus(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online CV+ conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train)
    T = len(y_all)

    y_hat_all = np.zeros(T)
    y_hat_all[:n_train] = yhat_train
    for t in range(n_train, T):
        y_hat_all[t] = estimator.predict(X_all[t : t + 1])[0]

    residuals = np.abs(y_all - y_hat_all)
    lower, upper = [], []
    for t in range(n_train, T):
        q = np.quantile(residuals[:t], 1 - alpha)
        lower.append(yhat_test[t - n_train] - q)
        upper.append(yhat_test[t - n_train] + q)

    return yhat_test, np.array(lower), np.array(upper)


def _predict_online_jackknife_ab(
    model_builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    n_bootstrap: int = 30,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Online Jackknife+ after Bootstrap (Online-J+aB)."""
    rng = np.random.default_rng(random_state)
    lowers, uppers, preds = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(X_train), len(X_train), replace=True)
        m = model_builder()
        m.fit(X_train[idx], y_train[idx])
        res = np.abs(y_train[idx] - m.predict(X_train[idx]))
        q = np.quantile(res, 1 - alpha)
        p = m.predict(X_test)
        preds.append(p)
        lowers.append(p - q)
        uppers.append(p + q)
    y_pred = np.mean(preds, axis=0)
    y_lower = np.min(lowers, axis=0)
    y_upper = np.max(uppers, axis=0)
    return y_pred, y_lower, y_upper


def _predict_cop(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    lr: float = 0.05,
    T_burnin: int = 50,
    scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conformal Optimistic Prediction (COP)."""
    estimator.fit(X_train, y_train)
    y_pred_test = estimator.predict(X_test).ravel()
    y_pred_train = estimator.predict(X_train).ravel()

    scores_train = np.abs(y_train - y_pred_train)
    init_q = np.quantile(scores_train, 1 - alpha)

    scores = np.abs(y_test - y_pred_test)
    T_test = len(scores)

    qs = np.zeros(T_test)
    qts = np.zeros(T_test)
    integrators = np.zeros(T_test)
    covereds = np.zeros(T_test)

    qs[0] = init_q
    qts[0] = init_q

    for t in range(T_test - 1):
        covereds[t] = 1 if qs[t] >= scores[t] else 0
        grad = alpha if covereds[t] else -(1 - alpha)
        if t < T_burnin:
            grad_i = 0.0
        else:
            window_s = scores[t - T_burnin : t]
            current_target = qts[t] - lr * grad
            grad_i = np.mean(window_s <= current_target) - (1 - alpha)

        integrator = -scale * grad_i
        qts[t + 1] = qts[t] - lr * grad
        integrators[t + 1] = lr * integrator
        qs[t + 1] = max(0.0, qts[t + 1] + integrators[t + 1])

    covereds[-1] = 1 if qs[-1] >= scores[-1] else 0
    y_lower = y_pred_test - qs
    y_upper = y_pred_test + qs
    return y_pred_test, y_lower, y_upper


def _predict_extreme(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
    gpd_threshold_quantile: float = 0.9,
    n_bootstrap: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extreme Conformal Prediction via Generalized Pareto Distribution."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train).ravel()
    yhat_test = estimator.predict(X_test).ravel()

    # Residuals of upper tail
    scores = y_train - yhat_train
    scores = scores[scores > 0]

    if len(scores) < 30:
        # Fallback to empirical quantile
        q_extreme = np.quantile(scores, 1 - alpha)
    else:
        u = np.quantile(scores, gpd_threshold_quantile)
        excesses = scores[scores > u] - u
        boot_q = []
        n_exc = len(excesses)
        n_scores = len(scores)

        for _ in range(n_bootstrap):
            res = np.random.choice(excesses, size=n_exc, replace=True)
            xi, _, sigma_gpd = genpareto.fit(res, floc=0)
            xi = max(1e-4, xi)
            q = u + (sigma_gpd / xi) * ((n_scores / (n_scores * alpha)) ** xi - 1)
            boot_q.append(q)

        q_extreme = np.quantile(boot_q, 0.95)

    y_lower = yhat_test - q_extreme
    y_upper = yhat_test + q_extreme
    return yhat_test, y_lower, y_upper


# AlphaNet PyTorch components
class AlphaNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def interval_size(scores, alpha_val, eps=1e-6):
    n = scores.numel()
    denominator = torch.clamp(alpha_val * (n + 1) - 1.0, min=eps)
    return 2.0 * scores.sum() / denominator


def _compute_scores_without_index(base_model, X_calib, y_calib, holdout_idx):
    loo_X = np.delete(X_calib, holdout_idx, axis=0)
    loo_y = np.delete(y_calib, holdout_idx, axis=0)
    return np.abs(base_model.predict(loo_X) - loo_y)


def _build_loo_feature_matrix(base_model, X_calib, y_calib):
    rows = []
    for idx in range(len(X_calib)):
        scores = _compute_scores_without_index(base_model, X_calib, y_calib, idx)
        rows.append([scores.sum()])
    return torch.tensor(np.asarray(rows, dtype=np.float32))


def _train_alpha_net(
    base_model,
    X_calib,
    y_calib,
    lambdas=(10.0, 20.0, 50.0),
    num_runs=3,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    alpha_clip=1e-3,
    seed=42,
    device="cpu",
):
    X_train_alpha = _build_loo_feature_matrix(base_model, X_calib, y_calib)
    all_results = {}
    for lambda_reg in lambdas:
        lambda_losses = []
        lambda_sizes = []
        lambda_alphas = []
        lambda_models = []
        for run_idx in range(num_runs):
            local_seed = seed + run_idx
            torch.manual_seed(local_seed)
            np.random.seed(local_seed)
            random.seed(local_seed)
            dataset = TensorDataset(X_train_alpha, torch.arange(len(X_train_alpha), dtype=torch.long))
            loader = DataLoader(dataset, batch_size=min(batch_size, len(X_train_alpha)), shuffle=True)
            alpha_net = AlphaNet(input_dim=X_train_alpha.shape[1]).to(device)
            optimizer = optim.Adam(alpha_net.parameters(), lr=learning_rate)
            all_losses = []
            all_sizes = []
            all_alphas = []
            for _ in range(epochs):
                epoch_losses = []
                epoch_sizes = []
                epoch_alphas = []
                alpha_net.train()
                for x_batch, idx_batch in loader:
                    x_batch = x_batch.to(device)
                    alpha_min = 1.0 / (len(X_calib) + 1.0) + 0.01
                    alpha_pred = torch.clamp(alpha_net(x_batch), min=max(alpha_clip, alpha_min), max=1.0 - alpha_clip)
                    batch_sizes = []
                    for j, idx in enumerate(idx_batch.tolist()):
                        scores_np = _compute_scores_without_index(base_model, X_calib, y_calib, idx)
                        scores_tensor = torch.tensor(scores_np, dtype=torch.float32, device=device)
                        batch_sizes.append(interval_size(scores_tensor, alpha_pred[j]))
                    batch_sizes = torch.stack(batch_sizes)
                    loss = (batch_sizes + lambda_reg * alpha_pred).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.item()))
                    epoch_sizes.append(float(batch_sizes.mean().item()))
                    epoch_alphas.append(float(alpha_pred.mean().item()))
                all_losses.append(float(np.mean(epoch_losses)))
                all_sizes.append(float(np.mean(epoch_sizes)))
                all_alphas.append(float(np.mean(epoch_alphas)))
            alpha_net.eval()
            lambda_losses.append(np.asarray(all_losses, dtype=np.float32))
            lambda_sizes.append(np.asarray(all_sizes, dtype=np.float32))
            lambda_alphas.append(np.asarray(all_alphas, dtype=np.float32))
            lambda_models.append(alpha_net)
        all_results[lambda_reg] = {
            "all_losses": np.vstack(lambda_losses),
            "all_sizes": np.vstack(lambda_sizes),
            "all_alphas": np.vstack(lambda_alphas),
            "models": lambda_models,
        }
    return all_results


def _select_best_alpha_net(all_results):
    best_lambda = None
    best_run_idx = None
    best_loss = float("inf")
    for lambda_reg, result_dict in all_results.items():
        final_losses = result_dict["all_losses"][:, -1]
        run_idx = int(np.argmin(final_losses))
        run_loss = float(final_losses[run_idx])
        if run_loss < best_loss:
            best_loss = run_loss
            best_lambda = lambda_reg
            best_run_idx = run_idx
    return best_lambda, best_run_idx, all_results[best_lambda]["models"][best_run_idx]


def _predict_alphanet(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    lambdas: tuple[float, ...] = (10.0, 20.0, 50.0),
    num_runs: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    alpha_clip: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AlphaNet / Adaptive Conformal Prediction (ACP)."""
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.35, random_state=seed
    )
    if len(X_calib) < 30:
        import warnings
        msg = f"Calibration dataset size ({len(X_calib)}) is too small (< 30) to support LOO training for AlphaNet/ACP. Falling back to empirical split conformal prediction."
        warnings.warn(msg, UserWarning)
        logging.warning(msg)
        return _predict_mfcs_split(
            estimator, X_train, y_train, X_test, alpha, calibration_size=0.35, random_state=seed
        )

    estimator.fit(X_fit, y_fit)
    all_results = _train_alpha_net(
        base_model=estimator,
        X_calib=X_calib,
        y_calib=y_calib,
        lambdas=lambdas,
        num_runs=num_runs,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        alpha_clip=alpha_clip,
        seed=seed,
        device=device,
    )
    best_lambda, best_run_idx, alpha_net = _select_best_alpha_net(all_results)

    calib_scores = np.abs(estimator.predict(X_calib) - y_calib)
    test_feature = torch.tensor([[calib_scores.sum()]], dtype=torch.float32, device=device)
    with torch.no_grad():
        alpha_hat = float(torch.clamp(alpha_net(test_feature), min=1e-3, max=1.0 - 1e-3).item())

    scores_tensor = torch.tensor(calib_scores, dtype=torch.float32, device=device)
    interval_width = float(interval_size(scores_tensor, torch.tensor(alpha_hat, device=device)).item())

    y_pred = estimator.predict(X_test).ravel()
    y_lower = y_pred - 0.5 * interval_width
    y_upper = y_pred + 0.5 * interval_width
    return y_pred, y_lower, y_upper


def _predict_normalized_cqr(
    lower_estimator,
    upper_estimator,
    median_estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    alpha: float,
    fit_ratio: float = 0.65,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalized (Multiplicative) CQR as used in the notebooks."""
    X_train_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
    y_train_np = y_train.to_numpy().ravel() if hasattr(y_train, "to_numpy") else np.asarray(y_train).ravel()
    X_test_np = X_test.to_numpy() if hasattr(X_test, "to_numpy") else np.asarray(X_test)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train_np, y_train_np, test_size=(1.0 - fit_ratio), random_state=random_state
    )

    lower_estimator.fit(X_fit, y_fit)
    upper_estimator.fit(X_fit, y_fit)
    median_estimator.fit(X_fit, y_fit)

    cal_lower = lower_estimator.predict(X_cal).ravel()
    cal_upper = upper_estimator.predict(X_cal).ravel()
    cal_pred = median_estimator.predict(X_cal).ravel()

    cal_U = np.maximum(cal_upper - cal_lower, np.finfo(float).eps)
    cal_scores = np.abs(cal_pred - y_cal) / cal_U

    n_cal = len(X_cal)
    qhat = np.quantile(cal_scores, np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)

    val_lower = lower_estimator.predict(X_test_np).ravel()
    val_upper = upper_estimator.predict(X_test_np).ravel()
    val_pred = median_estimator.predict(X_test_np).ravel()
    val_U = np.maximum(val_upper - val_lower, np.finfo(float).eps)

    y_lower = val_pred - val_U * qhat
    y_upper = val_pred + val_U * qhat
    y_pred = val_pred

    return y_pred, y_lower, y_upper


# =============================================================================
# MFCS Methods (Model-Free Conformal Selection)
# =============================================================================


def _predict_mfcs_split(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    calibration_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MFCS Split conformal prediction."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=calibration_size, shuffle=False, random_state=random_state
    )

    estimator.fit(X_fit, y_fit)
    y_cal_pred = estimator.predict(X_cal)
    yhat_test = estimator.predict(X_test)

    scores = np.abs(y_cal - y_cal_pred)
    q = np.quantile(scores, 1 - alpha, method="higher")

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


def _predict_mfcs_full(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MFCS Full conformal prediction."""
    estimator.fit(X_train, y_train)
    yhat_train = estimator.predict(X_train)
    yhat_test = estimator.predict(X_test)

    scores = np.abs(y_train - yhat_train)
    q = np.quantile(scores, 1 - alpha)

    y_lower = yhat_test - q
    y_upper = yhat_test + q
    return yhat_test, y_lower, y_upper


# =============================================================================
# Main Runner Functions
# =============================================================================


def _run_advanced_method(
    method_name: str,
    model_ids: list[str],
    model_params: dict[str, dict[str, Any]],
    bundle,
    alpha: float,
    decay: float,
    n_folds: int,
    n_bootstrap: int,
    calibration_size: float,
    random_state: int,
    cv: Any = "split",
    fit_ratio: float = 0.8,
    K: int = 5,
    gpd_threshold_quantile: float = 0.9,
    device: str = "cpu",
) -> PhaseResult:
    """Run a single advanced conformal method across all models."""
    metrics_rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.DataFrame] = {}

    X_train = bundle.X_train.to_numpy() if hasattr(bundle.X_train, "to_numpy") else np.asarray(bundle.X_train)
    y_train = _to_numpy(bundle.y_train)
    X_test = bundle.X_test.to_numpy() if hasattr(bundle.X_test, "to_numpy") else np.asarray(bundle.X_test)
    y_test = _to_numpy(bundle.y_test)

    for model_id in model_ids:
        params = dict(model_params.get(model_id, {}))

        def model_builder():
            return model_registry.build_estimator(
                model_id=model_id, phase=PHASE_CONFORMAL_ADVANCED, params=params
            )

        estimator = model_builder()

        backend = method_name
        try:
            if method_name == "nexcp_split":
                y_pred, y_lower, y_upper = _predict_nexcp_split(
                    estimator, X_train, y_train, X_test, alpha, decay
                )
            elif method_name == "nexcp_full":
                y_pred, y_lower, y_upper = _predict_nexcp_full(
                    estimator, X_train, y_train, X_test, alpha
                )
            elif method_name == "nexcp_jackknife_ab":
                y_pred, y_lower, y_upper = _predict_nexcp_jackknife_ab(
                    model_builder, X_train, y_train, X_test, alpha, n_bootstrap, random_state
                )
            elif method_name == "nexcp_cv_plus":
                y_pred, y_lower, y_upper = _predict_nexcp_cv_plus(
                    model_builder, X_train, y_train, X_test, alpha, n_folds, random_state
                )
            elif method_name == "online_split":
                y_pred, y_lower, y_upper = _predict_online_split(
                    estimator, X_train, y_train, X_test, alpha
                )
            elif method_name == "faci":
                y_pred, y_lower, y_upper = _predict_faci(
                    estimator, X_train, y_train, X_test, y_test, alpha
                )
            elif method_name == "mfcs_split":
                y_pred, y_lower, y_upper = _predict_mfcs_split(
                    estimator, X_train, y_train, X_test, alpha, calibration_size, random_state
                )
            elif method_name == "mfcs_full":
                y_pred, y_lower, y_upper = _predict_mfcs_full(
                    estimator, X_train, y_train, X_test, alpha
                )
            elif method_name == "cvplus":
                y_pred, y_lower, y_upper = _predict_puncc_cvplus(
                    estimator, bundle.X_train, bundle.y_train, bundle.X_test, alpha, K=K, random_state=random_state
                )
            elif method_name == "cqr":
                # CQR needs quantile models - build lower and upper
                lower_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=alpha / 2
                )
                upper_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=1 - alpha / 2
                )
                y_pred, y_lower, y_upper = _predict_puncc_cqr(
                    lower_est, upper_est, bundle.X_train, bundle.y_train, bundle.X_test, alpha, fit_ratio=fit_ratio, random_state=random_state
                )
            elif method_name in ("normalized_cqr", "ncqr"):
                lower_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=alpha / 2
                )
                upper_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=1 - alpha / 2
                )
                median_est = model_registry.build_estimator(
                    model_id=model_id, phase="quantile", params=params, quantile=0.5
                )
                y_pred, y_lower, y_upper = _predict_normalized_cqr(
                    lower_est, upper_est, median_est, bundle.X_train, bundle.y_train, bundle.X_test, alpha, fit_ratio=fit_ratio, random_state=random_state
                )
            elif method_name == "saocp":
                y_pred, y_lower, y_upper = _predict_saocp(
                    estimator, X_train, y_train, X_test, y_test, alpha
                )
            elif method_name == "sf_ogd":
                y_pred, y_lower, y_upper = _predict_sf_ogd(
                    estimator, X_train, y_train, X_test, y_test, alpha
                )
            elif method_name == "online_cvplus":
                y_pred, y_lower, y_upper = _predict_online_cv_plus(
                    estimator, X_train, y_train, X_test, y_test, alpha
                )
            elif method_name == "online_jackknife_ab":
                y_pred, y_lower, y_upper = _predict_online_jackknife_ab(
                    model_builder, X_train, y_train, X_test, alpha, n_bootstrap, random_state
                )
            elif method_name == "cop":
                y_pred, y_lower, y_upper = _predict_cop(
                    estimator, X_train, y_train, X_test, y_test, alpha
                )
            elif method_name == "extreme":
                y_pred, y_lower, y_upper = _predict_extreme(
                    estimator, X_train, y_train, X_test, y_test, alpha, gpd_threshold_quantile, n_bootstrap
                )
            elif method_name in ("alphanet", "acp"):
                y_pred, y_lower, y_upper = _predict_alphanet(
                    estimator, X_train, y_train, X_test, alpha, seed=random_state, device=device
                )
            else:
                raise ValueError(f"Unknown advanced conformal method '{method_name}'")

        except Exception as e:
            backend = "manual_fallback"
            import warnings
            msg = f"Conformal advanced method '{method_name}' failed on model '{model_id}': {str(e)}. Falling back to manual Split Conformal."
            warnings.warn(msg, UserWarning)
            logging.warning(msg)
            y_pred, y_lower, y_upper = _predict_mfcs_split(
                estimator, X_train, y_train, X_test, alpha, calibration_size, random_state
            )

        adv_mapping = {
            "nexcp_split": "NexCP-Split",
            "nexcp_full": "NexCP-Full",
            "nexcp_jackknife_ab": "NexCP-J+aB",
            "nexcp_cv_plus": "NexCP-CV+",
            "online_split": "Online-Split",
            "faci": "FACI",
            "mfcs_split": "MFCS-Split",
            "mfcs_full": "MFCS-Full",
            "cvplus": "CV+",
            "cqr": "CQR",
            "normalized_cqr": "Normalized-CQR",
            "saocp": "SAOCP",
            "sf_ogd": "SF-OGD",
            "online_cvplus": "Online-CV+",
            "online_jackknife_ab": "Online-J+aB",
            "cop": "COP",
            "extreme": "Extreme-CP",
            "alphanet": "AlphaNet",
        }
        strat_name = adv_mapping.get(method_name.lower(), method_name)

        y_true = np.asarray(y_test).ravel()
        pred_df = pd.DataFrame(
            {
                "sample_index": np.arange(len(y_true)),
                "strategy": strat_name,
                "y_true": y_true,
                "y_pred": y_pred,
                "ymin": y_lower,
                "ymax": y_upper,
                "width": y_upper - y_lower,
                "is_covered": ((y_true >= y_lower) & (y_true <= y_upper)).astype(int),
                "residual": y_true - y_pred,
                "y_lower": y_lower,
                "y_upper": y_upper,
                "backend": backend,
            }
        )
        predictions[model_id] = pred_df

        row = {"model": model_id, "method": method_name, "backend": backend, "alpha": alpha}
        row.update(regression_metrics(y_true=y_test, y_pred=y_pred))
        row.update(interval_metrics(y_true=y_test, y_lower=y_lower, y_upper=y_upper))
        from .metrics import cwc, ssc
        row["cwc"] = cwc(y_true=y_test, y_lower=y_lower, y_upper=y_upper, alpha=alpha)
        row["ssc"] = ssc(y_true=y_test, y_pred=y_pred, y_lower=y_lower, y_upper=y_upper)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    return PhaseResult(
        phase=f"{PHASE_CONFORMAL_ADVANCED}_{method_name}",
        predictions=predictions,
        metrics=metrics_df,
        params=model_params,
        artifacts=[],
    )


def run_conformal_advanced(
    data: Any,
    target_col: str,
    models: list[str] | tuple[str, ...] | None = None,
    params_source: Mapping[str, Any] | None = None,
    conformal_config: Mapping[str, Any] | None = None,
    output_config: OutputConfig | Mapping[str, Any] | None = None,
    split_config: SplitConfig | Mapping[str, Any] | None = None,
) -> ConformalResult:
    """Run advanced conformal prediction methods.

    Supported methods:
    - nexcp_split: NexCP Split with exponential weighting
    - nexcp_full: NexCP Full conformal
    - nexcp_jackknife_ab: NexCP Jackknife+ after Bootstrap
    - nexcp_cv_plus: NexCP Cross-Validation Plus
    - online_split: Online Split conformal
    - faci: Fully Adaptive Conformal Inference
    - mfcs_split: Model-Free Conformal Selection (Split)
    - mfcs_full: Model-Free Conformal Selection (Full)
    - cvplus: PUNCC CV+ (Cross-Validation Plus)
    - cqr: PUNCC CQR (Conformalized Quantile Regression)
    - normalized_cqr: Normalized (Multiplicative) CQR
    - saocp: Semi-Adaptive Online CP
    - sf_ogd: Scale-Free Online Gradient Descent
    - online_cvplus: Online CV+
    - online_jackknife_ab: Online Jackknife+ after Bootstrap
    - cop: Conformal Optimistic Prediction
    - extreme: Extreme CP via GPD tail fitting
    - alphanet: AlphaNet Adaptive Conformal Prediction

    Args:
        data: Input data (DataFrame, CSV path, or dict with train/test).
        target_col: Name of the target column.
        models: List of model IDs to use. Defaults to all supported models.
        params_source: Source for model parameters.
        conformal_config: Configuration dict with keys:
            - alpha: Miscoverage rate (default: 0.1)
            - methods: List of methods to run (default: all)
            - decay: Decay factor for NexCP (default: 0.99)
            - n_folds: Number of folds for CV methods (default: 5)
            - n_bootstrap: Bootstrap iterations (default: 30)
            - calibration_size: Calibration set proportion (default: 0.2)
            - random_state: Random seed (default: 42)
            - cv: cv strategy (default: "split")
            - fit_ratio: fit ratio (default: 0.8)
            - K: K parameter (default: 5)
            - gpd_threshold_quantile: GPD threshold quantile (default: 0.9)
            - device: device (default: "cpu")
        output_config: Output configuration.
        split_config: Train/test split configuration.

    Returns:
        ConformalResult with predictions and metrics for each method.
    """
    bundle = prepare_data_bundle(data=data, target_col=target_col, split_config=split_config)
    model_ids = model_registry.validate_models(models=models, phase=PHASE_CONFORMAL_ADVANCED)
    output_cfg = coerce_output_config(output_config)

    cfg = dict(conformal_config or {})
    alpha = float(cfg.get("alpha", DEFAULT_ALPHA))
    decay = float(cfg.get("decay", 0.99))
    n_folds = int(cfg.get("n_folds", 5))
    n_bootstrap = int(cfg.get("n_bootstrap", 30))
    calibration_size = float(cfg.get("calibration_size", 0.2))
    random_state = int(cfg.get("random_state", 42))
    cv = cfg.get("cv", "split")
    fit_ratio = float(cfg.get("fit_ratio", 0.8))
    K = int(cfg.get("K", 5))
    gpd_threshold_quantile = float(cfg.get("gpd_threshold_quantile", 0.9))
    device = str(cfg.get("device", "cpu"))

    default_methods = [
        "nexcp_split",
        "nexcp_full",
        "nexcp_jackknife_ab",
        "nexcp_cv_plus",
        "online_split",
        "faci",
        "mfcs_split",
        "mfcs_full",
        "cvplus",
        "cqr",
        "normalized_cqr",
        "saocp",
        "sf_ogd",
        "online_cvplus",
        "online_jackknife_ab",
        "cop",
        "extreme",
        "alphanet",
    ]
    methods = list(cfg.get("methods", default_methods))

    if not (0 < alpha < 1):
        raise ValueError("conformal alpha must satisfy 0 < alpha < 1")

    model_params, tuned_params = resolve_model_params(
        models=model_ids,
        params_source=params_source,
        X_train=bundle.X_train,
        y_train=bundle.y_train,
    )

    method_results: dict[str, PhaseResult] = {}
    for method_name in methods:
        method_results[method_name] = _run_advanced_method(
            method_name=method_name,
            model_ids=model_ids,
            model_params=model_params,
            bundle=bundle,
            alpha=alpha,
            decay=decay,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
            calibration_size=calibration_size,
            random_state=random_state,
            cv=cv,
            fit_ratio=fit_ratio,
            K=K,
            gpd_threshold_quantile=gpd_threshold_quantile,
            device=device,
        )

    result = ConformalResult(
        phase=PHASE_CONFORMAL_ADVANCED,
        methods=method_results,
        artifacts=[],
    )

    if output_cfg.output_dir is not None:
        output_dir = Path(output_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        chart_result = None
        if output_cfg.export_plots or output_cfg.embed_excel_charts or output_cfg.show_inline_plots:
            chart_result = generate_conformal_charts(
                conformal_result=result,
                output_cfg=output_cfg,
                output_dir=output_dir,
            )
        if output_cfg.export_plots and chart_result is not None:
            result.artifacts.extend(chart_result.image_paths)

        excel_path = output_dir / "conformal_advanced.xlsx"
        if output_cfg.export_excel:
            result.artifacts.append(write_conformal_excel(result, excel_path))
            if output_cfg.embed_excel_charts and chart_result is not None and chart_result.images_by_sheet:
                redirected_images = {}
                for sh_name, imgs in chart_result.images_by_sheet.items():
                    if sh_name.startswith("m_"):
                        redirected_images.setdefault(sh_name, []).extend(imgs)
                    else:
                        redirected_images.setdefault("all_pred_values", []).extend(imgs)
                embed_images_in_excel(workbook_path=excel_path, images_by_sheet=redirected_images)

        if output_cfg.save_json and tuned_params:
            result.artifacts.append(write_json(tuned_params, output_dir / "conformal_advanced_tuned_params.json"))

    return result
