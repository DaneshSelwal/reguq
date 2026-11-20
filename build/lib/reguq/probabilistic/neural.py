r"""Neural distributional regressors (need ``torch``): CARD and HCM.

CARD - Classification And Regression Diffusion (Han, Zheng & Zhou, NeurIPS 2022)
    A pre-trained mean model :math:`f_\phi(x)` is the prior mean. Forward
    process (paper Eq. 2):

    .. math:: y_t = \sqrt{\bar\alpha_t}\,y_0 + (1-\sqrt{\bar\alpha_t})\,f_\phi(x)
                    + \sqrt{1-\bar\alpha_t}\,\epsilon.

    :math:`\epsilon_\theta(x, y_t, f_\phi(x), t)` is trained with
    :math:`\|\epsilon - \epsilon_\theta\|^2`. Sampling starts at
    :math:`y_T \sim N(f_\phi(x), 1)` and follows the posterior
    :math:`q(y_{t-1}\mid y_t, \hat y_0, x)` with mean
    :math:`\gamma_0 \hat y_0 + \gamma_1 y_t + \gamma_2 f_\phi(x)` and variance
    :math:`\tilde\beta_t`, where

    .. math::
        \gamma_0 = \frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t},\;
        \gamma_1 = \frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t},\;
        \gamma_2 = 1 + \frac{(\sqrt{\bar\alpha_t}-1)(\sqrt{\alpha_t}+\sqrt{\bar\alpha_{t-1}})}{1-\bar\alpha_t},\;
        \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,

    and :math:`\hat y_0 = (y_t - (1-\sqrt{\bar\alpha_t})f_\phi - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}`.
    The network uses the paper's conditional-linear layers (128 units,
    softplus, timestep-embedding gates); targets are standardised.

HCM - Hyperspherical Confidence Mapping (Choi et al., ICLR 2026)
    The scalar target is embedded as :math:`y_{exp} = (y, y)`, with magnitude
    :math:`R = \|y_{exp}\|` and direction :math:`d = y_{exp}/R`. A two-head
    network predicts :math:`\hat R, \hat d`; loss
    :math:`e_R^2 + R^2\|e_d\|^2 + \lambda_{norm}(\|\hat d\|-1)^2`. Uncertainty
    :math:`\hat\sigma^2 = \hat R^2\,|\|\hat d\|^2 - 1|/(D-1)`, then a
    temperature :math:`s` is fitted on a **separate** calibration split.

    Limitation (documented, not hidden): for a scalar target the direction
    :math:`d` only encodes :math:`\mathrm{sign}(y)`, so the raw signal mostly
    reflects sign ambiguity near :math:`y \approx 0` (the target mean after
    standardisation). ``target_transform`` lets you choose the centring; for a
    guarantee, conformalize HCM's sigma with
    ``SplitConformalRegressor(score="normalized")``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from .._base import ArrayLike, as_1d_float
from .._utils import check_fitted, fit_clone, predict_1d, take_rows
from ..splits import split_indices
from ._base import BaseDistributionalRegressor, require
from ._dist import GaussianPrediction, SamplePrediction


def _torch() -> Any:
    return require("torch", "torch")


def _device(device: str) -> Any:
    torch = _torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ===================================================================== CARD
def _card_network(torch: Any, x_dim: int, n_steps: int, hidden: int) -> Any:
    nn = torch.nn

    class ConditionalLinear(nn.Module):  # type: ignore[name-defined]
        def __init__(self, num_in: int, num_out: int) -> None:
            super().__init__()
            self.lin = nn.Linear(num_in, num_out)
            self.embed = nn.Embedding(n_steps, num_out)
            self.embed.weight.data.uniform_()

        def forward(self, x: Any, t: Any) -> Any:
            return self.embed(t) * self.lin(x)

    class Guided(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.l1 = ConditionalLinear(x_dim + 2, hidden)
            self.l2 = ConditionalLinear(hidden, hidden)
            self.l3 = ConditionalLinear(hidden, hidden)
            self.out = nn.Linear(hidden, 1)
            self.act = nn.Softplus()

        def forward(self, x: Any, y_t: Any, f: Any, t: Any) -> Any:
            h = torch.cat([y_t, f, x], dim=1)
            h = self.act(self.l1(h, t))
            h = self.act(self.l2(h, t))
            h = self.act(self.l3(h, t))
            return self.out(h)

    return Guided()


class CARDRegressor(BaseDistributionalRegressor):
    """CARD diffusion regressor (see module docstring).

    Parameters
    ----------
    mean_estimator
        Regressor used as the pre-trained prior mean :math:`f_\\phi(x)`.
    n_steps
        Diffusion steps T (paper: 1000, linear betas 1e-4 .. 0.02).
    epochs, batch_size, lr
        Training of :math:`\\epsilon_\\theta` (Adam).
    n_samples
        Draws per test point returned by :meth:`predict_dist`.
    """

    def __init__(
        self,
        mean_estimator: Any,
        *,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        hidden: int = 128,
        epochs: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        n_samples: int = 100,
        device: str = "auto",
        random_state: int | None = 0,
    ) -> None:
        self.mean_estimator = mean_estimator
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_samples = n_samples
        self.device = device
        self.random_state = random_state

    def _schedule(self, torch: Any, dev: Any) -> dict[str, Any]:
        betas = torch.linspace(self.beta_start, self.beta_end, self.n_steps, device=dev)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        ab_prev = torch.cat([torch.ones(1, device=dev), ab[:-1]])
        return {"betas": betas, "alphas": alphas, "ab": ab, "ab_prev": ab_prev}

    def fit(self, X: ArrayLike, y: ArrayLike) -> CARDRegressor:
        torch = _torch()
        dev = _device(self.device)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        y_arr = as_1d_float(y)
        self.f_ = fit_clone(self.mean_estimator, X, y_arr)
        self.x_scaler_ = StandardScaler().fit(np.asarray(X, dtype=float))
        self.y_mean_, self.y_std_ = float(y_arr.mean()), float(y_arr.std() or 1.0)
        xs = torch.tensor(
            self.x_scaler_.transform(np.asarray(X, float)), dtype=torch.float32, device=dev
        )
        y0 = torch.tensor((y_arr - self.y_mean_) / self.y_std_, dtype=torch.float32, device=dev)[
            :, None
        ]
        f = torch.tensor(
            (predict_1d(self.f_, X) - self.y_mean_) / self.y_std_, dtype=torch.float32, device=dev
        )[:, None]
        sch = self._schedule(torch, dev)
        net = _card_network(torch, xs.shape[1], self.n_steps, self.hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        n = xs.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(self.random_state or 0)
        self.loss_history_ = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen).to(dev)
            tot = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                t = torch.randint(0, self.n_steps, (b.numel(),), device=dev)
                ab_t = sch["ab"][t][:, None]
                eps = torch.randn_like(y0[b])
                y_t = ab_t.sqrt() * y0[b] + (1 - ab_t.sqrt()) * f[b] + (1 - ab_t).sqrt() * eps
                loss = ((eps - net(xs[b], y_t, f[b], t)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss) * b.numel()
            self.loss_history_.append(tot / n)
        self.net_ = net.eval()
        return self

    def sample(
        self, X: ArrayLike, n_samples: int | None = None, seed: int | None = None
    ) -> np.ndarray:
        """Draws from the reverse process, shape ``(n, n_samples)``, in target units."""
        check_fitted(self, "net_")
        torch = _torch()
        dev = next(self.net_.parameters()).device
        m = int(n_samples or self.n_samples)
        g = torch.Generator(device=dev).manual_seed(self.random_state if seed is None else seed)
        xs = torch.tensor(
            self.x_scaler_.transform(np.asarray(X, float)), dtype=torch.float32, device=dev
        )
        f = torch.tensor(
            (predict_1d(self.f_, X) - self.y_mean_) / self.y_std_, dtype=torch.float32, device=dev
        )
        n = xs.shape[0]
        xs = xs.repeat_interleave(m, dim=0)
        f = f.repeat_interleave(m)[:, None]
        sch = self._schedule(torch, dev)
        y_t = f + torch.randn(f.shape, generator=g, device=dev)
        with torch.no_grad():
            for t in range(self.n_steps - 1, -1, -1):
                tt = torch.full((xs.shape[0],), t, device=dev, dtype=torch.long)
                ab_t, ab_prev = sch["ab"][t], sch["ab_prev"][t]
                a_t, b_t = sch["alphas"][t], sch["betas"][t]
                eps = self.net_(xs, y_t, f, tt)
                y0_hat = (y_t - (1 - ab_t.sqrt()) * f - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                if t == 0:
                    y_t = y0_hat
                    break
                g0 = b_t * ab_prev.sqrt() / (1 - ab_t)
                g1 = (1 - ab_prev) * a_t.sqrt() / (1 - ab_t)
                g2 = 1 + (ab_t.sqrt() - 1) * (a_t.sqrt() + ab_prev.sqrt()) / (1 - ab_t)
                var = (1 - ab_prev) / (1 - ab_t) * b_t
                mean = g0 * y0_hat + g1 * y_t + g2 * f
                y_t = mean + var.sqrt() * torch.randn(y_t.shape, generator=g, device=dev)
        out = y_t.cpu().numpy().reshape(n, m) * self.y_std_ + self.y_mean_
        return out

    def predict_dist(self, X: ArrayLike) -> SamplePrediction:
        return SamplePrediction(self.sample(X))


# ====================================================================== HCM
class HCMRegressor(BaseDistributionalRegressor):
    """Hyperspherical confidence mapping (see module docstring).

    Splits the training data into fit / early-stopping / temperature parts so
    the temperature is not tuned on the data used for early stopping.
    """

    def __init__(
        self,
        *,
        hidden: int = 64,
        n_layers: int = 3,
        embed_dim: int = 2,
        lam_norm: float = 0.0,
        epochs: int = 1000,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 50,
        val_size: float = 0.15,
        calib_size: float = 0.15,
        target_transform: str = "standardize",
        time_ordered: bool = False,
        device: str = "auto",
        random_state: int | None = 0,
    ) -> None:
        self.hidden = hidden
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.lam_norm = lam_norm
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_size = val_size
        self.calib_size = calib_size
        self.target_transform = target_transform
        self.time_ordered = time_ordered
        self.device = device
        self.random_state = random_state

    def _y_fwd(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_loc_) / self.y_scale_

    def _raw(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Mean and raw sigma in transformed target units."""
        torch = _torch()
        dev = next(self.net_.parameters()).device
        xs = torch.tensor(
            self.x_scaler_.transform(np.asarray(X, float)), dtype=torch.float32, device=dev
        )
        with torch.no_grad():
            h = self.net_["body"](xs)
            d = self.net_["d"](h)
            R = self.net_["R"](h).squeeze(1)
        d, R = d.cpu().numpy(), R.cpu().numpy()
        # y_exp = (y, ..., y): R = sqrt(D)|y| and d_k = sign(y)/sqrt(D), so R * d_k = y.
        mean = R * d.mean(axis=1)
        sig = np.abs(R) * np.sqrt(np.abs((d**2).sum(axis=1) - 1.0) / (self.embed_dim - 1))
        return mean, sig

    def fit(self, X: ArrayLike, y: ArrayLike) -> HCMRegressor:
        torch = _torch()
        dev = _device(self.device)
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        if self.embed_dim < 2:
            raise ValueError("embed_dim must be >= 2")
        y_arr = as_1d_float(y)
        n = y_arr.size
        rest, cal = split_indices(
            n, self.calib_size, time_ordered=self.time_ordered, random_state=self.random_state
        )
        tr_rel, val_rel = split_indices(
            rest.size,
            self.val_size / (1 - self.calib_size),
            time_ordered=self.time_ordered,
            random_state=self.random_state,
        )
        tr, val = rest[tr_rel], rest[val_rel]
        X_tr = np.asarray(take_rows(X, tr), float)
        self.x_scaler_ = StandardScaler().fit(X_tr)
        if self.target_transform == "standardize":
            self.y_loc_, self.y_scale_ = float(y_arr[tr].mean()), float(y_arr[tr].std() or 1.0)
        elif self.target_transform == "none":
            self.y_loc_, self.y_scale_ = 0.0, 1.0
        else:
            raise ValueError("target_transform must be 'standardize' or 'none'")

        def tensors(idx: np.ndarray) -> tuple[Any, Any]:
            xs = torch.tensor(
                self.x_scaler_.transform(np.asarray(take_rows(X, idx), float)),
                dtype=torch.float32,
                device=dev,
            )
            ys = torch.tensor(self._y_fwd(y_arr[idx]), dtype=torch.float32, device=dev)[:, None]
            return xs, ys.repeat(1, self.embed_dim)

        xt, yt = tensors(tr)
        xv, yv = tensors(val)
        nn = torch.nn
        layers: list[Any] = []
        d_in = xt.shape[1]
        for _ in range(self.n_layers):
            layers += [nn.Linear(d_in, self.hidden), nn.ReLU()]
            d_in = self.hidden
        net = nn.ModuleDict(
            {
                "body": nn.Sequential(*layers),
                "d": nn.Linear(self.hidden, self.embed_dim),
                "R": nn.Linear(self.hidden, 1),
            }
        ).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        def loss_fn(xb: Any, yb: Any) -> Any:
            h = net["body"](xb)
            d_hat, R_hat = net["d"](h), net["R"](h)
            R = yb.norm(dim=1, keepdim=True)
            d = yb / (R + 1e-8)
            e_R = ((R_hat - R) ** 2).squeeze(1)
            e_d = (R**2).squeeze(1) * ((d_hat - d) ** 2).sum(dim=1)
            pen = (d_hat.norm(dim=1) - 1.0) ** 2
            return (e_R + e_d + self.lam_norm * pen).mean()

        best, best_state, wait = np.inf, None, 0
        gen = torch.Generator(device="cpu").manual_seed(self.random_state or 0)
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(xt.shape[0], generator=gen).to(dev)
            for start in range(0, xt.shape[0], self.batch_size):
                b = perm[start : start + self.batch_size]
                loss = loss_fn(xt[b], yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss_fn(xv, yv))
            if v < best - 1e-7:
                best, wait = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net.eval()

        # Temperature on the held-out calibration split: match Gaussian
        # coverage at 1, 2, 3 sigma (68.27 / 95.45 / 99.73 %).
        mu_c, sig_c = self._raw(take_rows(X, cal))
        err = np.abs(self._y_fwd(y_arr[cal]) - mu_c)
        pos = sig_c[sig_c > 0]
        self.sigma_floor_ = 1e-3 * (np.median(pos) if pos.size else 1.0)
        sig_c = np.maximum(sig_c, self.sigma_floor_)
        ks = np.array([1.0, 2.0, 3.0])
        targets = 2 * norm.cdf(ks) - 1
        grid = np.geomspace(1e-3, 1e3, 2001)
        cover = (
            err[None, None, :] <= ks[None, :, None] * grid[:, None, None] * sig_c[None, None, :]
        ).mean(axis=2)
        self.temperature_ = float(grid[np.argmin(((cover - targets) ** 2).sum(axis=1))])
        return self

    def predict_dist(self, X: ArrayLike) -> GaussianPrediction:
        check_fitted(self, "temperature_")
        mu, sig = self._raw(X)
        sig = np.maximum(sig, self.sigma_floor_) * self.temperature_
        return GaussianPrediction(mu * self.y_scale_ + self.y_loc_, sig * self.y_scale_)
