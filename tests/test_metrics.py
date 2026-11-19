import numpy as np
import pytest
from scipy.stats import norm

from reguq import metrics as M


def test_coverage_no_broadcasting():
    y = np.array([0.0, 1.0, 2.0])
    interval = np.array([[-1.0, 0.5], [2.0, 3.0], [1.5, 2.5]])
    assert M.covered(y, interval).tolist() == [True, False, True]
    assert M.coverage(y, interval) == pytest.approx(2 / 3)
    # Column-vector bounds (the MAPIE (n,1) pitfall) must not broadcast to n x n.
    assert M.coverage(y, interval[:, :1], interval[:, 1:]) == pytest.approx(2 / 3)


def test_width_and_interval_score():
    y = np.array([0.0, 5.0])
    lo = np.array([-1.0, 0.0])
    hi = np.array([1.0, 2.0])
    assert M.mean_width(y, lo, hi) == 2.0
    # point 1 covered: 2; point 2 above by 3: 2 + 2/0.1*3 = 62 -> mean 32
    assert M.interval_score(y, lo, hi, alpha=0.1) == pytest.approx(32.0)


def test_cwc_penalises_undercoverage():
    y = np.linspace(0, 1, 100)
    full = np.column_stack([y - 1, y + 1])
    assert M.coverage_width_criterion(y, full, alpha=0.1) == pytest.approx(
        M.normalized_mean_width(y, full)
    )
    none = np.column_stack([y + 1, y + 2])
    assert M.coverage_width_criterion(y, none, alpha=0.1) > 1e10


def test_pinball():
    assert M.pinball_loss([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert M.pinball_loss([0.0], [1.0], 0.9) == pytest.approx(0.1)


def test_crps_gaussian_matches_sample_and_quantile():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    mu = rng.normal(size=50)
    sigma = rng.uniform(0.5, 2.0, size=50)
    exact = M.crps_gaussian(y, mu, sigma)
    samples = mu[:, None] + sigma[:, None] * rng.standard_normal((50, 20000))
    assert M.crps_sample(y, samples, fair=True) == pytest.approx(exact, rel=0.02)
    taus = (np.arange(1, 1001) - 0.5) / 1000
    qpred = mu[:, None] + sigma[:, None] * norm.ppf(taus)[None, :]
    assert M.crps_quantile(y, qpred, taus) == pytest.approx(exact, rel=0.01)


def test_crps_sample_pairwise_formula():
    x = np.array([[0.0, 1.0, 3.0]])
    y = np.array([1.0])
    naive = np.mean(np.abs(x - 1.0)) - 0.5 * np.mean(np.abs(x[0][:, None] - x[0][None, :]))
    assert M.crps_sample(y, x) == pytest.approx(naive)


def test_quantile_grid_weights():
    taus = (np.arange(1, 11) - 0.5) / 10
    assert np.allclose(M.quantile_grid_weights(taus), 0.1)
    w = M.quantile_grid_weights([0.05, 0.1, 0.5, 0.9, 0.95])
    assert w.sum() == pytest.approx(1.0)


def test_gaussian_nll():
    assert M.gaussian_nll([0.0], [0.0], [1.0]) == pytest.approx(0.5 * np.log(2 * np.pi))
    with pytest.raises(ValueError):
        M.gaussian_nll([0.0], [0.0], [0.0])


@pytest.mark.statistical
def test_pit_calibrated_forecast_is_uniform():
    rng = np.random.default_rng(3)
    mu = rng.normal(size=20000)
    sigma = rng.uniform(0.5, 2, size=20000)
    y = mu + sigma * rng.standard_normal(20000)
    err = M.calibration_error(M.pit_gaussian(y, mu, sigma))
    assert err["mce"] < 0.02
    assert M.ence(y, mu, sigma) < 0.05
    # An overconfident forecast (sigma halved) must be flagged.
    assert M.ence(y, mu, sigma / 2) > 0.5


def test_calibration_curve_excludes_trivial_endpoints():
    p, _ = M.calibration_curve(np.linspace(0, 1, 11))
    assert p.min() > 0 and p.max() < 1


def test_interval_report_keys():
    y = np.linspace(0, 1, 50)
    rep = M.interval_report(y, np.column_stack([y - 0.1, y + 0.1]), alpha=0.1)
    assert rep["coverage"] == 1.0
    assert set(rep) >= {"coverage", "mean_width", "interval_score", "cwc", "ssc"}


def test_point_report():
    rep = M.point_report([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert rep["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert rep["bias"] == pytest.approx(1 / 3)
