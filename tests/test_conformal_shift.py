import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from conftest import make_linear
from reguq.conformal import (
    ACI,
    AdaptiveCoveragePolicyRegressor,
    ExtremeValueConformalRegressor,
    NexCPRegressor,
    OnlineConformalRegressor,
    SplitConformalRegressor,
    exponential_weights,
    gpd_tail_quantile,
)
from reguq.metrics import coverage

ALPHA = 0.1


def drifting_series(n, rng, shift_at=None, shift=4.0):
    """Linear model whose noise scale jumps at `shift_at` (distribution shift)."""
    X = rng.uniform(-2, 2, size=(n, 2))
    sd = np.ones(n)
    if shift_at is not None:
        sd[shift_at:] = shift
    y = X @ np.array([1.0, -2.0]) + sd * rng.standard_normal(n)
    return X, y


def test_exponential_weights_newest_largest():
    w = exponential_weights(4, 0.5)
    assert np.allclose(w, [0.5**4, 0.5**3, 0.5**2, 0.5])


def test_nexcp_rho_one_equals_split():
    rng = np.random.default_rng(0)
    X, y = make_linear(400, rng)
    a = NexCPRegressor(LinearRegression(), rho=1.0, calib_size=100).fit(X[:300], y[:300])
    b = SplitConformalRegressor(LinearRegression(), calib_size=100, time_ordered=True).fit(
        X[:300], y[:300]
    )
    assert np.allclose(a.predict_interval(X[300:], ALPHA), b.predict_interval(X[300:], ALPHA))


@pytest.mark.statistical
def test_online_nexcp_adapts_to_shift_better_than_static():
    rng = np.random.default_rng(1)
    X, y = drifting_series(3000, rng, shift_at=1500)
    static = SplitConformalRegressor(LinearRegression(), calib_size=300, time_ordered=True).fit(
        X[:1000], y[:1000]
    )
    online = OnlineConformalRegressor(
        LinearRegression(), method="nexcp", alpha=ALPHA, method_params={"rho": 0.99}, calib_size=300
    ).fit(X[:1000], y[:1000])
    cov_static = coverage(y[1000:], static.predict_interval(X[1000:], ALPHA))
    cov_online = coverage(y[1000:], online.predict_interval_online(X[1000:], y[1000:]))
    assert cov_online > cov_static + 0.1
    assert abs(cov_online - (1 - ALPHA)) < 0.03


@pytest.mark.parametrize("method", ["aci", "dtaci", "sf_ogd", "saocp", "nexcp"])
def test_online_methods_are_causal(method):
    """Changing y_t must not change the interval at t (no look-ahead)."""
    rng = np.random.default_rng(2)
    X, y = drifting_series(700, rng)
    m1 = OnlineConformalRegressor(LinearRegression(), method=method, alpha=ALPHA).fit(
        X[:400], y[:400]
    )
    m2 = OnlineConformalRegressor(LinearRegression(), method=method, alpha=ALPHA).fit(
        X[:400], y[:400]
    )
    y2 = y[400:].copy()
    t = 150
    y2[t] += 1e3
    iv1 = m1.predict_interval_online(X[400:], y[400:])
    iv2 = m2.predict_interval_online(X[400:], y2)
    assert np.allclose(iv1[: t + 1], iv2[: t + 1])
    assert not np.allclose(iv1[t + 1 :], iv2[t + 1 :])


@pytest.mark.statistical
def test_aci_long_run_error_bound():
    """Gibbs & Candès (2021) Prop. 4.1 holds for any score sequence, even adversarial."""
    rng = np.random.default_rng(3)
    gamma, T = 0.01, 5000
    scores = np.concatenate([rng.exponential(1, T // 2), rng.exponential(10, T // 2)])
    tr = ACI(ALPHA, gamma=gamma)
    tr.warm_start(rng.exponential(1, 200))
    errs = []
    for s in scores:
        errs.append(s > tr.predict())
        tr.update(s)
    bound = (max(ALPHA, 1 - ALPHA) + gamma) / (gamma * T)
    assert abs(np.mean(errs) - ALPHA) <= bound + 1e-12


@pytest.mark.statistical
@pytest.mark.parametrize("method", ["aci", "dtaci", "sf_ogd", "saocp"])
def test_online_methods_long_run_coverage_under_shift(method):
    rng = np.random.default_rng(4)
    X, y = drifting_series(5000, rng, shift_at=2500)
    m = OnlineConformalRegressor(
        LinearRegression(), method=method, alpha=ALPHA, calib_size=300
    ).fit(X[:1000], y[:1000])
    iv = m.predict_interval_online(X[1000:], y[1000:])
    assert abs(coverage(y[1000:], iv) - (1 - ALPHA)) < 0.03


def test_sf_ogd_moves_in_right_direction():
    from reguq.conformal import ScaleFreeOGD

    tr = ScaleFreeOGD(ALPHA, scale=10.0)
    tr.q = 5.0
    tr.update(1.0)  # covered -> shrink
    assert tr.q < 5.0
    q = tr.q
    tr.update(100.0)  # miss -> grow
    assert tr.q > q


@pytest.mark.statistical
def test_gpd_tail_quantile_exponential():
    rng = np.random.default_rng(5)
    s = rng.exponential(size=20000)
    for alpha in (1e-3, 1e-4):
        q = gpd_tail_quantile(s, alpha, threshold=0.95)["q"]
        assert q == pytest.approx(-np.log(alpha), rel=0.15)


def test_gpd_falls_back_to_conformal_above_threshold():
    s = np.random.default_rng(6).exponential(size=1000)
    out = gpd_tail_quantile(s, 0.2, threshold=0.9)
    assert out["method"] == "conformal"


def test_extreme_regressor_two_sided():
    rng = np.random.default_rng(7)
    X, y = make_linear(3000, rng)
    m = ExtremeValueConformalRegressor(LinearRegression(), threshold=0.9, calib_size=2000).fit(X, y)
    iv = m.predict_interval(X[:5], alpha=1e-4)
    mu = m.predict(X[:5])
    assert np.allclose(mu - iv[:, 0], iv[:, 1] - mu)
    assert np.isfinite(iv).all()


def test_acp_default_lambda_targets_alpha():
    rng = np.random.default_rng(8)
    X, y = make_linear(1500, rng)
    m = AdaptiveCoveragePolicyRegressor(LinearRegression(), alpha=ALPHA, calib_size=500).fit(X, y)
    assert m.alpha_tilde_ == pytest.approx(ALPHA, rel=0.02)
    # radius formula
    S = m.calib_scores_
    assert m.radius(0.2) == pytest.approx(S.sum() / (501 * 0.2 - 1))


def test_acp_target_size_bisection():
    rng = np.random.default_rng(9)
    X, y = make_linear(1500, rng)
    m = AdaptiveCoveragePolicyRegressor(LinearRegression(), target_size=10.0, calib_size=500).fit(
        X, y
    )
    assert m._loo_size(m.policy_) == pytest.approx(10.0, rel=1e-3)


@pytest.mark.statistical
def test_acp_posthoc_guarantee():
    """E[ 1{miss} / alpha_tilde ] <= 1 (Gauthier et al. Prop. 2.2)."""
    rng = np.random.default_rng(10)
    vals = []
    for _ in range(300):
        X, y = make_linear(500, rng)
        m = AdaptiveCoveragePolicyRegressor(LinearRegression(), alpha=0.2, calib_size=100).fit(
            X[:300], y[:300]
        )
        iv = m.predict_interval(X[300:])
        miss = 1 - coverage(y[300:], iv)
        vals.append(miss / m.alpha_tilde_)
    assert np.mean(vals) <= 1.0 + 0.05
