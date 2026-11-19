import numpy as np
import pytest
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from conftest import make_linear
from reguq.conformal import SplitConformalRegressor
from reguq.metrics import coverage
from reguq.probabilistic import (
    GaussianPrediction,
    IBUGRegressor,
    SamplePrediction,
    StdEstimator,
)


def test_gaussian_prediction_interval_and_quantiles():
    g = GaussianPrediction(np.zeros(3), np.ones(3) * 2)
    iv = g.interval(0.05)
    assert np.allclose(iv[:, 1], 2 * norm.ppf(0.975))
    assert np.allclose(g.quantiles([0.5])[:, 0], 0)
    with pytest.raises(ValueError):
        GaussianPrediction(np.zeros(2), np.zeros(2))


def test_sample_prediction_matches_gaussian_in_limit():
    rng = np.random.default_rng(0)
    mu, sd = rng.normal(size=20), rng.uniform(1, 2, 20)
    g = GaussianPrediction(mu, sd)
    s = SamplePrediction(g.sample(20000, random_state=1))
    y = mu + sd * rng.standard_normal(20)
    assert s.crps(y) == pytest.approx(g.crps(y), rel=0.03)
    assert np.allclose(s.std(), sd, rtol=0.05)


@pytest.mark.statistical
def test_ibug_std_is_not_square_rooted_and_is_calibrated():
    rng = np.random.default_rng(1)
    X, y = make_linear(2500, rng, hetero=True)
    m = IBUGRegressor(
        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0), random_state=0
    ).fit(X[:2000], y[:2000])
    d = m.predict_dist(X[2000:])
    # Noise sd ranges 0.5..2.5; a sqrt'ed std would sit near 1 everywhere.
    assert np.corrcoef(np.abs(X[2000:, 0]), d.sigma)[0, 1] > 0.3
    cov = coverage(y[2000:], d.interval(0.1))
    assert 0.8 <= cov <= 0.97
    assert m.k_ >= 3


def test_ibug_rejects_non_tree_models():
    rng = np.random.default_rng(2)
    X, y = make_linear(200, rng)
    with pytest.raises(TypeError):
        IBUGRegressor(LinearRegression()).fit(X, y)


@pytest.mark.statistical
def test_conformalized_distributional_sigma_is_valid():
    """A distributional model's std as the normalized score scale keeps the CP guarantee."""
    rng = np.random.default_rng(3)
    covs = []
    for _ in range(20):
        X, y = make_linear(1600, rng, hetero=True)
        base = IBUGRegressor(
            RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=0),
            random_state=0,
        ).fit(X[:800], y[:800])
        cp = SplitConformalRegressor(
            base, score="normalized", scale_estimator=StdEstimator(base), prefit=True
        ).fit(X[800:1100], y[800:1100])
        covs.append(coverage(y[1100:], cp.predict_interval(X[1100:], 0.1)))
    assert np.mean(covs) >= 0.9 - 0.015


# ------------------------------------------------------------- optional backends
def test_ngboost_sigma_is_std():
    pytest.importorskip("ngboost")
    from reguq.probabilistic import NGBoostRegressor

    rng = np.random.default_rng(4)
    X, y = make_linear(1500, rng)
    y = y * 4.0  # noise sd 4: std -> ~4, variance would be ~16, sqrt(std) ~2
    m = NGBoostRegressor(
        n_estimators=500, learning_rate=0.05, random_state=0, early_stopping_rounds=20
    ).fit(X[:1000], y[:1000], X_val=X[1000:1250], y_val=y[1000:1250])
    d = m.predict_dist(X[1250:])
    assert 2.6 < np.median(d.sigma) < 6.5


def test_pgbm_returns_std():
    pytest.importorskip("pgbm")
    from reguq.probabilistic import PGBMRegressor

    rng = np.random.default_rng(5)
    X, y = make_linear(800, rng)
    m = PGBMRegressor(random_state=0).fit(X[:600], y[:600])
    d = m.predict_dist(X[600:])
    assert d.sigma.shape == (200,) and (d.sigma > 0).all()


def test_card_learns_noise_scale():
    pytest.importorskip("torch")
    from reguq.probabilistic import CARDRegressor

    rng = np.random.default_rng(6)
    X, y = make_linear(600, rng)
    y = y * 3.0  # noise sd 3: the v1.2 notebook bug produced sd ~ 1 regardless
    m = CARDRegressor(LinearRegression(), n_steps=100, epochs=150, n_samples=200, device="cpu").fit(
        X[:500], y[:500]
    )
    d = m.predict_dist(X[500:])
    assert 1.5 < np.median(d.std()) < 6.0


def test_hcm_runs_and_uses_separate_calibration():
    pytest.importorskip("torch")
    from reguq.probabilistic import HCMRegressor

    rng = np.random.default_rng(7)
    X, y = make_linear(600, rng)
    m = HCMRegressor(epochs=80, patience=20, device="cpu").fit(X[:500], y[:500])
    d = m.predict_dist(X[500:])
    assert d.mu.shape == (100,) and m.temperature_ > 0


def test_treeffuser_samples_shape():
    pytest.importorskip("treeffuser")
    from reguq.probabilistic import TreeffuserRegressor

    rng = np.random.default_rng(8)
    X, y = make_linear(400, rng)
    m = TreeffuserRegressor(n_samples=30, n_steps=20).fit(X[:300], y[:300])
    d = m.predict_dist(X[300:])
    assert d.samples.shape == (100, 30)
