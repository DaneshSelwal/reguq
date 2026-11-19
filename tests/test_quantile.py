import numpy as np
import pytest

from reguq.conformal import (
    conformal_lower_quantile,
    conformal_quantile,
    conformal_rank,
    weighted_conformal_quantile,
)


def test_rank_formula():
    # ceil((n+1)(1-alpha))
    assert conformal_rank(9, 0.1) == 9
    assert conformal_rank(99, 0.1) == 90  # (100)(0.9) = 90 exactly, despite float error
    assert conformal_rank(10, 0.1) == 10  # ceil(9.9)


def test_quantile_exact_order_statistic():
    scores = np.arange(1, 101, dtype=float)  # 1..100, n=100
    # k = ceil(101 * 0.9) = 91 -> 91st smallest = 91
    assert conformal_quantile(scores, 0.1) == 91.0
    # order of input must not matter
    rng = np.random.default_rng(0)
    assert conformal_quantile(rng.permutation(scores), 0.1) == 91.0


def test_quantile_infinite_for_small_n():
    # n=5, alpha=0.1: k = ceil(6*0.9)=6 > 5 -> +inf (not the max score)
    assert conformal_quantile([1, 2, 3, 4, 5], 0.1) == np.inf


def test_lower_quantile():
    v = np.arange(1, 101, dtype=float)
    # floor(0.1 * 101) = 10 -> 10th smallest
    assert conformal_lower_quantile(v, 0.1) == 10.0
    assert conformal_lower_quantile([1.0, 2.0, 3.0], 0.1) == -np.inf


def test_invalid_inputs():
    with pytest.raises(ValueError):
        conformal_quantile([], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, np.nan], 0.1)
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 1.0)


def test_weighted_uniform_matches_unweighted():
    rng = np.random.default_rng(1)
    for n in (10, 57, 200):
        s = rng.normal(size=n)
        for alpha in (0.05, 0.1, 0.3):
            assert weighted_conformal_quantile(s, np.ones(n), alpha) == conformal_quantile(s, alpha)


def test_weighted_includes_test_mass():
    # Two scores with weight 1 each plus test mass 1: each atom has 1/3.
    # Cumulative: 1/3, 2/3 -> never reaches 0.9 -> +inf.
    assert weighted_conformal_quantile([1.0, 2.0], [1.0, 1.0], 0.1) == np.inf
    # With alpha=0.5 the 2nd score (cum 2/3 >= 0.5) is chosen, no interpolation.
    assert weighted_conformal_quantile([1.0, 2.0], [1.0, 1.0], 0.5) == 2.0


def test_weighted_downweights_old_points():
    # Old huge scores with tiny weight should barely matter.
    s = np.concatenate([np.full(50, 100.0), np.linspace(0, 1, 50)])
    w = np.concatenate([np.full(50, 1e-6), np.ones(50)])
    q = weighted_conformal_quantile(s, w, 0.1)
    assert q <= 1.0


@pytest.mark.statistical
@pytest.mark.parametrize("n", [19, 100, 500])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_guarantee(n, alpha):
    """P(S_{n+1} <= q) in [1-alpha, 1-alpha + 1/(n+1)] for exchangeable scores."""
    rng = np.random.default_rng(12345 + n)
    trials = 20000
    s = rng.exponential(size=(trials, n + 1))
    q = np.array([conformal_quantile(row[:-1], alpha) for row in s])
    cov = np.mean(s[:, -1] <= q)
    se = np.sqrt(0.25 / trials)
    assert cov >= 1 - alpha - 4 * se
    assert cov <= 1 - alpha + 1 / (n + 1) + 4 * se
