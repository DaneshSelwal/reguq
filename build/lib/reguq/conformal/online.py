r"""Online (sequential) conformal prediction under distribution shift.

All methods here are **strictly causal**: the interval for time ``t`` is
computed from scores observed before ``t``; only afterwards is
:math:`y_t` revealed and used to update the state. Each method tracks a
radius :math:`q_t` for the absolute-residual score
:math:`s_t = |y_t - \hat\mu(x_t)|`, giving :math:`\hat C_t = \hat\mu(x_t) \pm q_t`.

Methods
-------
``ACI``
    Adaptive conformal inference, Gibbs & Candès (2021):
    :math:`\alpha_{t+1} = \alpha_t + \gamma(\alpha - \mathrm{err}_t)`,
    :math:`q_t = \hat Q_t(1-\alpha_t)` (empirical quantile of past scores).
    Guarantees :math:`|\tfrac1T\sum_t \mathrm{err}_t - \alpha|
    \le \tfrac{\max(\alpha_1, 1-\alpha_1)+\gamma}{\gamma T}` for any sequence.
``DtACI`` (a.k.a. FACI)
    Dynamically-tuned ACI, Gibbs & Candès (2024, JMLR; arXiv:2208.08401):
    exponentially re-weights ACI experts with step sizes
    :math:`\gamma_k = 0.001\cdot 2^k` using the pinball loss of
    :math:`\beta_t`. Constants follow the authors' reference implementation
    (Bhatnagar et al. 2023 code).
``ScaleFreeOGD``
    Online gradient descent on the pinball loss with scale-free step
    :math:`D/\sqrt{3\sum_s g_s^2}` (Orabona & Pál 2016; Bhatnagar et al. 2023).
``SAOCP``
    Strongly adaptive online conformal prediction, Bhatnagar, Wang, Xiong &
    Bai (2023, ICML): SF-OGD experts on geometric lifetimes aggregated by
    coin betting. Port of the Apache-2.0 Salesforce reference
    implementation (``online_conformal/saocp.py``).
``NexCPOnline``
    NexCP (Barber et al. 2023) with the calibration set growing as labels
    arrive and weights :math:`\rho^{age}`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import logsumexp

from .._base import ArrayLike, BaseIntervalRegressor, as_1d_float, as_interval, check_alpha
from .._utils import check_fitted, fit_clone, predict_1d
from ..splits import train_calib_split
from ._quantile import weighted_conformal_quantile


def _empirical_quantile(scores: list[float] | np.ndarray, level: float) -> float:
    """Plain empirical quantile ``inf{s : F_n(s) >= level}`` (inverted CDF)."""
    if len(scores) == 0:
        return float(np.inf)
    level = min(max(level, 0.0), 1.0)
    return float(np.quantile(np.asarray(scores, dtype=float), level, method="inverted_cdf"))


def _pinball(y: Any, yhat: Any, q: float) -> Any:
    return np.maximum(q * (y - yhat), (1 - q) * (yhat - y))


def _pinball_grad(y: float, yhat: float, q: float) -> float:
    """d/d(yhat) of the pinball loss at level ``q``."""
    return float(-q * (y > yhat) + (1 - q) * (y < yhat))


class OnlineTracker:
    """Base class: tracks the radius ``q_t`` of a symmetric interval."""

    def __init__(self, alpha: float) -> None:
        self.alpha = check_alpha(alpha)
        self.scores: list[float] = []

    def warm_start(self, scores: np.ndarray) -> None:
        """Seed with calibration scores (default: history only, no state updates)."""
        self.scores.extend(float(s) for s in scores)

    def predict(self) -> float:
        raise NotImplementedError

    def update(self, score: float) -> None:
        raise NotImplementedError


class ACI(OnlineTracker):
    """Adaptive conformal inference (Gibbs & Candès 2021).

    Parameters
    ----------
    gamma
        Step size. Larger adapts faster but makes widths noisier.
    window
        Use only the last ``window`` scores for :math:`\\hat Q_t` (None = all).
    """

    def __init__(self, alpha: float, gamma: float = 0.005, window: int | None = None) -> None:
        super().__init__(alpha)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        self.window = window
        self.alpha_t = self.alpha

    def _hist(self) -> list[float]:
        return self.scores[-self.window :] if self.window else self.scores

    def predict(self) -> float:
        if self.alpha_t <= 0:
            return float(np.inf)  # full real line
        if self.alpha_t >= 1:
            return 0.0  # (essentially) empty set
        return _empirical_quantile(self._hist(), 1.0 - self.alpha_t)

    def update(self, score: float) -> None:
        err = float(score > self.predict())
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        self.scores.append(float(score))


class DtACI(OnlineTracker):
    """Dynamically-tuned ACI / FACI (Gibbs & Candès 2024)."""

    def __init__(self, alpha: float, gammas: ArrayLike | None = None, interval: int = 100) -> None:
        super().__init__(alpha)
        self.gammas = (
            0.001 * 2.0 ** np.arange(8) if gammas is None else as_1d_float(gammas, "gammas")
        )
        self.k = self.gammas.size
        self.I = int(interval)
        self.sigma = 1.0 / (2 * self.I)
        a = self.alpha
        denom = ((1 - a) ** 2 * a**3 + a**2 * (1 - a) ** 3) / 3
        self.eta = np.sqrt(3 / self.I) * np.sqrt((np.log(self.I * self.k) + 2) / denom)
        self.alphas = np.full(self.k, a)
        self.log_w = np.zeros(self.k)

    @property
    def alpha_t(self) -> float:
        p = np.exp(self.log_w - logsumexp(self.log_w))
        return float(p @ self.alphas)

    def predict(self) -> float:
        return _empirical_quantile(self.scores, 1.0 - self.alpha_t)

    def update(self, score: float) -> None:
        hist = np.asarray(self.scores)
        if hist.size > int(np.floor(1 / self.alpha)):
            beta = float(np.mean(hist >= score))
            losses = _pinball(beta, self.alphas, self.alpha)
            wbar = self.log_w - self.eta * losses
            total = logsumexp(wbar)
            mixed = np.logaddexp(np.log1p(-self.sigma) + wbar, np.log(self.sigma / self.k) + total)
            self.log_w = mixed - logsumexp(mixed)
            err = (self.alphas > beta).astype(float)
            self.alphas = np.clip(self.alphas + self.gammas * (self.alpha - err), 0.0, 1.0)
        self.scores.append(float(score))


FACI = DtACI


class ScaleFreeOGD(OnlineTracker):
    """Scale-free online gradient descent on the pinball loss (Bhatnagar et al. 2023).

    ``scale`` (D) defaults to ``sqrt(3) * max(calibration scores)``; the
    calibration scores are replayed through :meth:`update` to initialise
    :math:`q`, as in the reference implementation.
    """

    def __init__(self, alpha: float, scale: float | None = None) -> None:
        super().__init__(alpha)
        self.scale = scale
        self.q = 0.0
        self.grad_norm = 0.0

    def warm_start(self, scores: np.ndarray) -> None:
        if self.scale is None:
            self.scale = 1.0 if len(scores) == 0 else float(np.max(np.abs(scores)) * np.sqrt(3))
        for s in scores:
            self.update(float(s))

    def predict(self) -> float:
        return self.q

    def update(self, score: float) -> None:
        if self.scale is None:
            self.scale = 1.0
        grad = _pinball_grad(abs(score), self.q, 1 - self.alpha)
        self.grad_norm += grad**2
        if self.grad_norm != 0:
            self.q = max(0.0, self.q - self.scale / np.sqrt(3 * self.grad_norm) * grad)
        self.scores.append(float(score))


class _SAOCPExpert:
    def __init__(self, t: int, scale: float, alpha: float, q0: float, g: int) -> None:
        self.scale = scale
        self.base_lr = scale / np.sqrt(3)
        self.alpha = alpha
        self.q = q0
        self.grad_norm = 0.0
        u = 0
        while t % 2 == 0:
            t //= 2
            u += 1
        self.lifetime = g * 2**u
        self.z = 0.0
        self.wz = 0.0
        self.s_t = 0

    @property
    def expired(self) -> bool:
        return self.s_t > self.lifetime

    @property
    def w(self) -> float:
        return 0.0 if self.s_t == 0 else self.z / self.s_t * (1 + self.wz)

    def update(self, s: float, meta_loss: float) -> None:
        w = self.w
        loss = _pinball(s, self.q, 1 - self.alpha)
        g = (meta_loss - loss) / self.scale / max(self.alpha, 1 - self.alpha)
        g = float(np.clip(g, -1.0 * (w > 0), 1.0))
        self.z += g
        self.wz += g * w
        self.s_t += 1
        grad = _pinball_grad(s, self.q, 1 - self.alpha)
        self.grad_norm += grad**2
        if self.grad_norm != 0:
            self.q = max(0.0, self.q - self.base_lr / np.sqrt(self.grad_norm) * grad)


class SAOCP(OnlineTracker):
    """Strongly adaptive online conformal prediction (Bhatnagar et al. 2023)."""

    def __init__(self, alpha: float, scale: float | None = None, lifetime: int = 8) -> None:
        super().__init__(alpha)
        self.scale = scale
        self.lifetime = lifetime
        self.t = 1
        self.experts: dict[int, _SAOCPExpert] = {}

    def _p(self) -> dict[int, float]:
        prior = {t: 1.0 / (t**2 * (1 + np.floor(np.log2(t)))) for t in self.experts}
        z = sum(prior.values())
        prior = {t: v / z for t, v in prior.items()}
        p = {t: prior[t] * max(0.0, e.w) for t, e in self.experts.items()}
        sp = sum(p.values())
        return {t: v / sp for t, v in p.items()} if sp > 0 else prior

    def warm_start(self, scores: np.ndarray) -> None:
        if self.scale is None:
            self.scale = 1.0 if len(scores) == 0 else float(np.max(np.abs(scores)) * np.sqrt(3))
        for s in scores:
            self.update(float(s))

    def predict(self) -> float:
        if not self.experts:
            return 0.0
        p = self._p()
        return float(sum(p[t] * e.q for t, e in self.experts.items()))

    def update(self, score: float) -> None:
        if self.scale is None:
            self.scale = 1.0
        s = abs(float(score))
        q_hat = self.predict()
        for t in [t for t, e in self.experts.items() if e.expired]:
            self.experts.pop(t)
        self.experts[self.t] = _SAOCPExpert(self.t, self.scale, self.alpha, q_hat, self.lifetime)
        meta_loss = float(_pinball(s, self.predict(), 1 - self.alpha))
        for e in self.experts.values():
            e.update(s, meta_loss)
        self.t += 1
        self.scores.append(s)


class NexCPOnline(OnlineTracker):
    """NexCP with a growing calibration set and weights ``rho**age`` (age 1 = newest)."""

    def __init__(self, alpha: float, rho: float = 0.99, window: int | None = None) -> None:
        super().__init__(alpha)
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must lie in (0, 1]")
        self.rho = rho
        self.window = window

    def predict(self) -> float:
        hist = self.scores[-self.window :] if self.window else self.scores
        if not hist:
            return float(np.inf)
        n = len(hist)
        w = self.rho ** (n - np.arange(n, dtype=float))
        return weighted_conformal_quantile(hist, w, self.alpha)

    def update(self, score: float) -> None:
        self.scores.append(float(score))


_METHODS = {
    "aci": ACI,
    "dtaci": DtACI,
    "faci": DtACI,
    "sf_ogd": ScaleFreeOGD,
    "saocp": SAOCP,
    "nexcp": NexCPOnline,
}


class OnlineConformalRegressor(BaseIntervalRegressor):
    """Wrap a regressor with an online conformal method.

    The model is fitted once on the proper-training part; the calibration
    part (the end of the training period) seeds the tracker. Test points are
    then processed in order with :meth:`predict_interval_online`.

    Parameters
    ----------
    estimator
        Scikit-learn style regressor.
    method
        One of ``"aci"``, ``"dtaci"`` (= ``"faci"``), ``"sf_ogd"``, ``"saocp"``, ``"nexcp"``.
    alpha
        Target long-run miscoverage.
    method_params
        Extra keyword arguments for the tracker (e.g. ``{"gamma": 0.01}``).
    """

    def __init__(
        self,
        estimator: Any,
        *,
        method: str = "aci",
        alpha: float = 0.1,
        method_params: dict[str, Any] | None = None,
        calib_size: float | int = 0.25,
    ) -> None:
        self.estimator = estimator
        self.method = method
        self.alpha = alpha
        self.method_params = method_params
        self.calib_size = calib_size

    def _new_tracker(self) -> OnlineTracker:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {sorted(_METHODS)}")
        return _METHODS[self.method](self.alpha, **(self.method_params or {}))

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_calib: ArrayLike | None = None,
        y_calib: ArrayLike | None = None,
    ) -> OnlineConformalRegressor:
        if X_calib is None:
            s = train_calib_split(X, y, self.calib_size, time_ordered=True)
            X_tr, y_tr, X_cal, y_cal = s.X_train, s.y_train, s.X_calib, s.y_calib
        else:
            X_tr, y_tr, X_cal, y_cal = X, as_1d_float(y), X_calib, as_1d_float(y_calib)
        self.estimator_ = fit_clone(self.estimator, X_tr, y_tr)
        self.tracker_ = self._new_tracker()
        self.tracker_.warm_start(np.abs(y_cal - predict_1d(self.estimator_, X_cal)))
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_fitted(self, "tracker_")
        return predict_1d(self.estimator_, X)

    def predict_interval(self, X: ArrayLike, alpha: float | None = None) -> np.ndarray:
        """Interval from the *current* state, without consuming labels."""
        check_fitted(self, "tracker_")
        if alpha is not None and alpha != self.alpha:
            raise ValueError("online methods are tied to the alpha given at construction")
        mu = self.predict(X)
        q = self.tracker_.predict()
        return as_interval(mu - q, mu + q)

    def predict_interval_online(self, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Process ``(X, y)`` sequentially: interval at ``t`` first, then reveal ``y_t``.

        Records ``radius_history_`` (and ``alpha_history_`` for ACI/DtACI).
        The tracker state carries over between calls.
        """
        check_fitted(self, "tracker_")
        y_arr = as_1d_float(y)
        mu = predict_1d(self.estimator_, X)
        radii = np.empty(y_arr.size)
        has_alpha = hasattr(self.tracker_, "alpha_t")
        alphas = np.empty(y_arr.size) if has_alpha else None
        for t in range(y_arr.size):
            if alphas is not None:
                alphas[t] = getattr(self.tracker_, "alpha_t")  # noqa: B009
            radii[t] = self.tracker_.predict()
            self.tracker_.update(abs(y_arr[t] - mu[t]))
        self.radius_history_ = radii
        self.alpha_history_ = alphas
        return as_interval(mu - radii, mu + radii)

    def update(self, X: ArrayLike, y: ArrayLike) -> None:
        """Reveal labels for already-predicted points (in order) without returning intervals."""
        check_fitted(self, "tracker_")
        mu = predict_1d(self.estimator_, X)
        for yt, mt in zip(as_1d_float(y), mu, strict=True):
            self.tracker_.update(abs(yt - mt))
