import numpy as np
from experiment_autopsy.cuped import cuped_adjust


def test_cuped_no_reduction_when_covariate_uncorrelated():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 10000)
    x = rng.normal(0, 1, 10000)  # independent of y
    _, theta, var_reduction = cuped_adjust(y, x)
    assert abs(theta) < 0.05
    assert var_reduction < 5  # essentially no reduction, allowing sampling noise


def test_cuped_variance_reduction_matches_rho_squared():
    rng = np.random.default_rng(1)
    rho = 0.6
    n = 20000
    cov = [[1, rho], [rho, 1]]
    z = rng.multivariate_normal([0, 0], cov, size=n)
    x, y = z[:, 0], z[:, 1]
    _, theta, var_reduction = cuped_adjust(y, x)
    # theoretical variance reduction for jointly normal (x,y) is rho^2
    assert abs(var_reduction - rho ** 2 * 100) < 3  # within 3pp


def test_cuped_never_increases_variance():
    rng = np.random.default_rng(2)
    for rho in [0.0, 0.2, 0.5, 0.8]:
        cov = [[1, rho], [rho, 1]]
        z = rng.multivariate_normal([0, 0], cov, size=5000)
        _, _, var_reduction = cuped_adjust(z[:, 1], z[:, 0])
        assert var_reduction >= -1  # allow tiny negative from sampling noise only
