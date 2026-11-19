import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

from conftest import make_linear
from reguq.conformal import ConformalizedQuantileRegressor
from reguq.metrics import pinball_loss
from reguq.quantile import (
    MultiQuantileRegressor,
    crossing_rate,
    rearrange_quantiles,
    with_quantile,
)


def test_rearrangement_removes_crossing_and_never_hurts_pinball():
    rng = np.random.default_rng(0)
    taus = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = rng.normal(size=500)
    q = np.sort(rng.normal(size=(500, 5)), axis=1) + rng.normal(scale=0.8, size=(500, 5))
    assert crossing_rate(q) > 0
    r = rearrange_quantiles(q)
    assert crossing_rate(r) == 0
    loss = lambda m: sum(pinball_loss(y, m[:, k], t) for k, t in enumerate(taus))  # noqa: E731
    assert loss(r) <= loss(q) + 1e-12


def test_multi_quantile_interval_and_interpolation():
    rng = np.random.default_rng(1)
    X, y = make_linear(600, rng, hetero=True)
    m = MultiQuantileRegressor(
        QuantileRegressor(alpha=0.0, solver="highs"), quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).fit(X, y)
    q = m.predict_quantiles(X)
    assert crossing_rate(q) == 0
    iv = m.predict_interval(X, 0.1)
    assert np.allclose(iv[:, 0], q[:, 0]) and np.allclose(iv[:, 1], q[:, 4])
    iv2 = m.predict_interval(X, 0.2)  # 0.1 / 0.9 interpolated
    assert np.all(iv2[:, 0] >= iv[:, 0] - 1e-9)
    with pytest.raises(ValueError):
        m.predict_interval(X, 0.05)  # 0.025 outside the grid


def test_quantile_models_wrap_into_cqr():
    rng = np.random.default_rng(2)
    X, y = make_linear(800, rng)
    est = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0)
    cqr = ConformalizedQuantileRegressor(est, alpha=0.1, random_state=0).fit(X[:600], y[:600])
    assert cqr.lower_.alpha == pytest.approx(0.05) and cqr.upper_.alpha == pytest.approx(0.95)
    assert cqr.predict_interval(X[600:]).shape == (200, 2)


def test_qrf_captures_noise():
    pytest.importorskip("quantile_forest")
    from reguq.quantile import QuantileForestRegressor

    rng = np.random.default_rng(3)
    X, y = make_linear(1500, rng)
    m = QuantileForestRegressor(n_estimators=100, random_state=0).fit(X[:1000], y[:1000])
    iv = m.predict_interval(X[1000:], 0.1)
    width = np.mean(iv[:, 1] - iv[:, 0])
    assert width > 2.0  # noise sd 1 -> true 90% width ~3.3; tree-spread width would be << 1


def test_tabnet_quantile_via_factory():
    pytest.importorskip("pytorch_tabnet")
    from reguq.quantile import TabNetRegressor

    rng = np.random.default_rng(4)
    X, y = make_linear(500, rng)
    lo = with_quantile(TabNetRegressor(max_epochs=30, patience=5), 0.1).fit(X, y)
    hi = with_quantile(TabNetRegressor(max_epochs=30, patience=5), 0.9).fit(X, y)
    assert np.mean(hi.predict(X) > lo.predict(X)) > 0.8
