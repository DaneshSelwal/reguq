r"""Adaptive coverage policies, ACP (Gauthier, Bach & Jordan, AISTATS 2026; arXiv:2510.04318).

Uses the soft-rank conformal e-value
:math:`E = S_{test} / \big(\tfrac{1}{n+1}(\sum_i S_i + S_{test})\big)` and the set
:math:`\{y : E(y) < 1/\tilde\alpha\}`. For the absolute-residual score this is
the interval :math:`\hat\mu(x) \pm q` with

.. math:: q = \frac{\sum_{i=1}^n S_i}{(n+1)\tilde\alpha - 1}, \qquad \tilde\alpha > \tfrac{1}{n+1}.

Because e-values give *post-hoc* validity, :math:`\tilde\alpha` may be chosen
from the data (here, by a learned policy) while keeping (Prop. 2.2)

.. math:: \mathbb E\Big[\frac{P(Y_{test} \notin \hat C^{\tilde\alpha} \mid \tilde\alpha)}{\tilde\alpha}\Big] \le 1.

This is **not** a :math:`1-\alpha` marginal guarantee; for a fixed
:math:`\tilde\alpha=\alpha` it reduces to :math:`P(Y \in \hat C) \ge 1-\alpha`
(Markov), which is more conservative than split conformal.

Policy training (paper Eq. 7-9): leave-one-out over the calibration set, with
pseudo-calibration sums :math:`T_j = \sum_{i\ne j} S_i`, minimising

.. math:: \frac1n \sum_j \frac{2T_j}{n\tilde\alpha_\theta(T_j) - 1} + \lambda\,\tilde\alpha_\theta(T_j).

In regression the policy's only input is :math:`T_j` (the test summary is
empty), so the optimal policy is nearly constant. ``policy="closed_form"``
returns the exact minimiser over constant policies,
:math:`\tilde\alpha^* = (1 + \sqrt{2\bar T n / \lambda})/n`;
``policy="mlp"`` trains the paper's network (one hidden layer of 32 ReLU,
sigmoid output, Adam 1e-3, batch 32, 200 epochs; needs ``torch``).

Choosing :math:`\lambda`
    :math:`\lambda` has the units of the interval size, so a fixed number
    (e.g. 20) is meaningless across datasets - with residuals in the hundreds
    it drives :math:`\tilde\alpha \to 1`. reguq offers (i) ``target_size``:
    bisection on :math:`\lambda` to hit a mean leave-one-out width (paper,
    Algorithm 2), or (ii) by default, the :math:`\lambda` whose constant
    optimum equals the reference ``alpha``:
    :math:`\lambda_\alpha = 2\bar T n/(n\alpha - 1)^2`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, predict_1d
from ..splits import train_calib_split


def _closed_form_alpha(t_bar: float, n: int, lam: float) -> float:
    a = (1.0 + np.sqrt(2.0 * t_bar * n / lam)) / n
    return float(np.clip(a, (1.0 + 1e-9) / n, 1.0))


class AdaptiveCoveragePolicyRegressor(BaseIntervalRegressor):
    """ACP with the absolute-residual score.

    Parameters
    ----------
    estimator
        Scikit-learn style regressor.
    alpha
        Reference miscoverage; sets the default ``lam`` (see module docs).
    lam
        Trade-off between size and :math:`\\tilde\\alpha`. Overrides ``alpha``.
    target_size
        Desired mean interval width (full width, target units); ``lam`` is
        found by bisection. Overrides ``lam`` and ``alpha``.
    policy
        ``"closed_form"`` (default, no torch needed) or ``"mlp"``.
    """

    def __init__(
        self,
        estimator: Any,
        *,
        alpha: float = 0.1,
        lam: float | None = None,
        target_size: float | None = None,
        policy: str = "closed_form",
        hidden: int = 32,
        epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        calib_size: float | int = 0.25,
        time_ordered: bool = False,
        random_state: Any = None,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.lam = lam
        self.target_size = target_size
        self.policy = policy
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.calib_size = calib_size
        self.time_ordered = time_ordered
        self.random_state = random_state

    # ------------------------------------------------------------ policies
    def _train_policy(self, lam: float) -> Any:
        """Return a callable ``T -> alpha_tilde`` trained with trade-off ``lam``."""
        n, T = self.n_calib_, self.loo_sums_
        if self.policy == "closed_form":
            a = _closed_form_alpha(float(T.mean()), n, lam)
            return lambda t: np.full(np.shape(t), a, dtype=float)
        if self.policy != "mlp":
            raise ValueError("policy must be 'closed_form' or 'mlp'")
        try:
            import torch
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("policy='mlp' needs torch: pip install 'reguq[torch]'") from e
        torch.manual_seed(0 if self.random_state is None else int(self.random_state))
        scale = float(T.mean())  # normalise the input and the loss to O(1)
        net = torch.nn.Sequential(
            torch.nn.Linear(1, self.hidden), torch.nn.ReLU(), torch.nn.Linear(self.hidden, 1)
        )
        with torch.no_grad():
            net[-1].bias.fill_(3.0)  # start near alpha_tilde ~ 1, as in the paper
        lo = 1.0 / n
        x = torch.tensor(T / scale, dtype=torch.float32).unsqueeze(1)
        t = torch.tensor(T, dtype=torch.float32)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        gen = torch.Generator().manual_seed(0)

        def alpha_of(inp: Any) -> Any:
            # (1/n, 1): keeps the size term finite by construction.
            return lo + (1.0 - lo) * torch.sigmoid(net(inp)).squeeze(1)

        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                a = alpha_of(x[b])
                loss = ((2 * t[b] / (n * a - 1) + lam * a) / scale).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        def policy(tval: Any) -> np.ndarray:
            inp = torch.tensor(np.atleast_1d(tval) / scale, dtype=torch.float32).unsqueeze(1)
            with torch.no_grad():
                return alpha_of(inp).numpy().astype(float)

        return policy

    def _loo_size(self, policy: Any) -> float:
        a = policy(self.loo_sums_)
        return float(np.mean(2 * self.loo_sums_ / (self.n_calib_ * a - 1)))

    def _select_lambda(self) -> float:
        n, t_bar = self.n_calib_, float(self.loo_sums_.mean())
        if self.target_size is not None:
            if self.target_size <= 0:
                raise ValueError("target_size must be positive")
            lo, hi = np.log(1e-10 * t_bar), np.log(1e10 * t_bar)
            for _ in range(60 if self.policy == "closed_form" else 25):
                mid = 0.5 * (lo + hi)
                size = self._loo_size(self._train_policy(float(np.exp(mid))))
                # larger lambda -> smaller alpha_tilde -> wider sets
                if size > self.target_size:
                    hi = mid
                else:
                    lo = mid
            return float(np.exp(0.5 * (lo + hi)))
        if self.lam is not None:
            if self.lam <= 0:
                raise ValueError("lam must be positive")
            return float(self.lam)
        a = check_alpha(self.alpha)
        if n * a <= 1:
            raise ValueError(f"need n_calib * alpha > 1 (n_calib={n}, alpha={a})")
        return 2.0 * t_bar * n / (n * a - 1.0) ** 2

    # ----------------------------------------------------------------- API
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_calib: ArrayLike | None = None,
        y_calib: ArrayLike | None = None,
    ) -> AdaptiveCoveragePolicyRegressor:
        if X_calib is None:
            s = train_calib_split(
                X,
                y,
                self.calib_size,
                time_ordered=self.time_ordered,
                random_state=self.random_state,
            )
            X_tr, y_tr, X_cal, y_cal = s.X_train, s.y_train, s.X_calib, s.y_calib
        else:
            X_tr, y_tr, X_cal, y_cal = X, as_1d_float(y), X_calib, as_1d_float(y_calib)
        self.estimator_ = fit_clone(self.estimator, X_tr, y_tr)
        S = np.abs(y_cal - predict_1d(self.estimator_, X_cal))
        self.calib_scores_ = S
        self.n_calib_ = S.size
        self.loo_sums_ = S.sum() - S
        self.lam_ = self._select_lambda()
        self.policy_ = self._train_policy(self.lam_)
        self.alpha_tilde_ = float(self.policy_(np.array([S.sum()]))[0])
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_fitted(self, "policy_")
        return predict_1d(self.estimator_, X)

    def radius(self, alpha_tilde: float) -> float:
        """Half-width :math:`\\sum S_i / ((n+1)\\tilde\\alpha - 1)` (inf if not positive)."""
        check_fitted(self, "policy_")
        denom = (self.n_calib_ + 1) * alpha_tilde - 1.0
        return float(np.inf) if denom <= 0 else float(self.calib_scores_.sum() / denom)

    def predict_interval(self, X: ArrayLike, alpha: float | None = None) -> np.ndarray:
        """Interval at the learned :math:`\\tilde\\alpha` (default) or at a fixed ``alpha``."""
        check_fitted(self, "policy_")
        a = self.alpha_tilde_ if alpha is None else check_alpha(alpha)
        mu = self.predict(X)
        q = self.radius(a)
        return as_interval(mu - q, mu + q)
