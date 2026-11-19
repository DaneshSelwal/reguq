import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge

from conftest import make_linear
from reguq.conformal import (
    ConformalizedQuantileRegressor,
    CrossConformalRegressor,
    JackknifePlusAfterBootstrapRegressor,
    SplitConformalRegressor,
)
from reguq.metrics import coverage
from reguq.quantile import with_quantile

ALPHA = 0.1


def _mean_coverage(make_model, reps, n_train, n_test=300, hetero=False, seed=0):
    rng = np.random.default_rng(seed)
    covs, widths = [], []
    for _ in range(reps):
        X, y = make_linear(n_train + n_test, rng, hetero=hetero)
        m = make_model().fit(X[:n_train], y[:n_train])
        iv = m.predict_interval(X[n_train:], ALPHA)
        covs.append(coverage(y[n_train:], iv))
        widths.append(np.mean(iv[:, 1] - iv[:, 0]))
    return float(np.mean(covs)), float(np.mean(widths))


@pytest.mark.statistical
def test_split_absolute_coverage_bounds():
    n_train, calib = 300, 100
    cov, _ = _mean_coverage(
        lambda: SplitConformalRegressor(LinearRegression(), calib_size=calib, random_state=1),
        reps=300,
        n_train=n_train,
    )
    tol = 0.01
    assert 1 - ALPHA - tol <= cov <= 1 - ALPHA + 1 / (calib + 1) + tol


@pytest.mark.statistical
def test_split_normalized_is_valid_and_adaptive():
    cov, _ = _mean_coverage(
        lambda: SplitConformalRegressor(
            LinearRegression(),
            score="normalized",
            scale_estimator=GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=0),
            calib_size=150,
            random_state=1,
        ),
        reps=60,
        n_train=500,
        hetero=True,
    )
    assert cov >= 1 - ALPHA - 0.015
    rng = np.random.default_rng(5)
    X, y = make_linear(600, rng, hetero=True)
    m = SplitConformalRegressor(
        LinearRegression(),
        score="normalized",
        calib_size=150,
        scale_estimator=GradientBoostingRegressor(random_state=0),
    ).fit(X, y)
    w = np.diff(m.predict_interval(X, ALPHA), axis=1).ravel()
    # width should grow with |x0| (the noise driver)
    assert np.corrcoef(np.abs(X[:, 0]), w)[0, 1] > 0.5


def test_split_small_calibration_gives_infinite_interval():
    rng = np.random.default_rng(0)
    X, y = make_linear(20, rng)
    m = SplitConformalRegressor(LinearRegression(), calib_size=5).fit(X, y)
    assert np.isinf(m.predict_interval(X[:3], 0.1)).all()


def test_prefit_and_explicit_calibration_agree():
    rng = np.random.default_rng(0)
    X, y = make_linear(200, rng)
    base = LinearRegression().fit(X[:100], y[:100])
    a = SplitConformalRegressor(base, prefit=True).fit(X[100:150], y[100:150])
    b = SplitConformalRegressor(LinearRegression()).fit(X[:100], y[:100], X[100:150], y[100:150])
    assert np.allclose(a.predict_interval(X[150:]), b.predict_interval(X[150:]))


@pytest.mark.statistical
def test_cqr_coverage_and_real_quantiles():
    est = QuantileRegressor(alpha=0.0, solver="highs")
    cov, _ = _mean_coverage(
        lambda: ConformalizedQuantileRegressor(est, alpha=ALPHA, calib_size=100, random_state=2),
        reps=80,
        n_train=300,
        hetero=True,
    )
    assert 1 - ALPHA - 0.015 <= cov <= 1 - ALPHA + 1 / 101 + 0.015
    # The two quantile models must differ (the notebook bug used one mean model twice).
    rng = np.random.default_rng(3)
    X, y = make_linear(400, rng, hetero=True)
    m = ConformalizedQuantileRegressor(est, alpha=ALPHA).fit(X, y)
    raw = m.predict_quantiles(X)
    assert np.all(raw[:, 1] - raw[:, 0] > 0)


@pytest.mark.statistical
def test_cqr_asymmetric_controls_each_tail():
    rng = np.random.default_rng(4)
    lo_miss, hi_miss = [], []
    for _ in range(60):
        X, y = make_linear(700, rng, hetero=True)
        m = ConformalizedQuantileRegressor(
            QuantileRegressor(alpha=0.0, solver="highs"),
            alpha=ALPHA,
            symmetric=False,
            calib_size=150,
            random_state=0,
        ).fit(X[:400], y[:400])
        iv = m.predict_interval(X[400:])
        lo_miss.append(np.mean(y[400:] < iv[:, 0]))
        hi_miss.append(np.mean(y[400:] > iv[:, 1]))
    assert np.mean(lo_miss) <= ALPHA / 2 + 0.015
    assert np.mean(hi_miss) <= ALPHA / 2 + 0.015


def test_with_quantile_maps_backends():
    g = with_quantile(GradientBoostingRegressor(), 0.9)
    assert g.loss == "quantile" and g.alpha == 0.9
    h = with_quantile(HistGradientBoostingRegressor(), 0.1)
    assert h.loss == "quantile" and h.quantile == 0.1
    with pytest.raises(TypeError):
        with_quantile(Ridge(), 0.9)  # 'alpha' there is regularisation, not a quantile


@pytest.mark.statistical
@pytest.mark.parametrize("method", ["plus", "minmax"])
def test_cv_plus_coverage(method):
    cov, _ = _mean_coverage(
        lambda: CrossConformalRegressor(
            LinearRegression(), n_splits=5, method=method, random_state=0
        ),
        reps=150,
        n_train=150,
    )
    assert cov >= 1 - ALPHA - 0.015


def test_jackknife_plus_loo_matches_definition():
    rng = np.random.default_rng(7)
    X, y = make_linear(30, rng)
    m = CrossConformalRegressor(LinearRegression(), n_splits="loo").fit(X, y)
    x0 = X[:1]
    # brute force Jackknife+
    lows, highs = [], []
    for i in range(30):
        keep = np.arange(30) != i
        mi = LinearRegression().fit(X[keep], y[keep])
        r = abs(y[i] - mi.predict(X[i : i + 1])[0])
        mu = mi.predict(x0)[0]
        lows.append(mu - r)
        highs.append(mu + r)
    k_lo = int(np.floor(0.2 * 31))
    k_hi = int(np.ceil(0.8 * 31))
    iv = m.predict_interval(x0, 0.2)
    assert iv[0, 0] == pytest.approx(np.sort(lows)[k_lo - 1])
    assert iv[0, 1] == pytest.approx(np.sort(highs)[k_hi - 1])


@pytest.mark.statistical
def test_jab_coverage():
    cov, _ = _mean_coverage(
        lambda: JackknifePlusAfterBootstrapRegressor(
            LinearRegression(), n_bootstraps=20, random_state=0
        ),
        reps=80,
        n_train=150,
    )
    assert cov >= 1 - ALPHA - 0.02


def test_estimators_are_sklearn_clone_safe():
    from sklearn.base import clone

    for est in [
        SplitConformalRegressor(LinearRegression(), calib_size=0.3),
        ConformalizedQuantileRegressor(GradientBoostingRegressor(), alpha=0.2),
        CrossConformalRegressor(LinearRegression(), n_splits=3),
        JackknifePlusAfterBootstrapRegressor(LinearRegression(), n_bootstraps=5),
    ]:
        c = clone(est)
        assert c.get_params(deep=False).keys() == est.get_params(deep=False).keys()
