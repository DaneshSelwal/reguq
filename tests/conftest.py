import numpy as np
import pytest


def make_linear(n, rng, hetero=False, d=3):
    X = rng.uniform(-2, 2, size=(n, d))
    beta = np.arange(1, d + 1, dtype=float)
    noise_sd = 0.5 + np.abs(X[:, 0]) if hetero else 1.0
    y = X @ beta + noise_sd * rng.standard_normal(n)
    return X, y


@pytest.fixture
def linear_data():
    return make_linear
