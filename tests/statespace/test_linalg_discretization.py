import numpy as np
import pytest

from pysip.statespace import discretization


@pytest.mark.parametrize("n", [2, 3])
def test_state_and_input_for_real_eigenvalues(n):
    rng = np.random.default_rng(42)
    eigenvalues = -rng.uniform(0.1, 1.0, n)
    vectors, _ = np.linalg.qr(rng.random((n, n)))
    A = vectors @ np.diag(eigenvalues) @ vectors.T
    B = rng.random((n, 2))
    dt = 0.4
    Ad, B0, B1 = discretization.state_input(A, B, dt, 1, "analytic")

    expected = vectors @ np.diag(np.exp(eigenvalues * dt)) @ vectors.T
    np.testing.assert_allclose(Ad, expected, atol=1e-14)
    np.testing.assert_allclose(A @ B0, (Ad - np.eye(n)) @ B, atol=1e-14)
    np.testing.assert_allclose(A @ B1, B0 - dt * B, atol=1e-14)


@pytest.mark.parametrize(
    "A, expected",
    [
        (np.array([[0.0, 1.0], [1.0, 0.0]]),
         np.array([[np.cosh(1.0), np.sinh(1.0)],
                   [np.sinh(1.0), np.cosh(1.0)]])),
        (np.array([[-1.0, 1.0], [0.0, -1.0]]),
         np.exp(-1.0) * np.array([[1.0, 1.0], [0.0, 1.0]])),
        (-np.eye(3), np.exp(-1.0) * np.eye(3)),
        (np.array([[0.0, -1.0], [1.0, 0.0]]),
         np.array([[np.cos(1.0), -np.sin(1.0)],
                   [np.sin(1.0), np.cos(1.0)]])),
    ],
    ids=["equal-diagonals", "defective-repeated-eigenvalue", "repeated-eigenvalues", "complex-eigenvalues"],
)
def test_state_adversarial_matrices(A, expected):
    np.testing.assert_allclose(discretization.state(A), expected, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("order_hold", [0, 1])
def test_singular_state_input_exponential(n, order_hold):
    A = np.zeros((n, n))
    B = np.arange(1.0, 2 * n + 1).reshape(n, 2)
    dt = 0.4
    Ad, B0, B1 = discretization.state_input(A, B, dt, order_hold)
    np.testing.assert_allclose(Ad, np.eye(n))
    np.testing.assert_allclose(B0, dt * B)
    np.testing.assert_allclose(B1, dt**2 / 2 * B if order_hold else np.zeros_like(B))
    with pytest.raises(np.linalg.LinAlgError):
        discretization.state_input(A, B, dt, order_hold, "analytic")


def test_long_scale_gp_diffusion_against_quadrature():
    """The Mauna Loa trend increment must not subtract million-scale variances."""
    from scipy.integrate import quad_vec
    from scipy.linalg import expm

    from pysip.statespace import Matern52

    model = Matern52([
        {"name": "mscale", "value": 2112.742111978681},
        {"name": "lscale", "value": 481.8830148206853},
        {"name": "sigv", "value": 0.215438432959732},
    ])
    model.update()
    rate = model.Q.T @ model.Q
    for dt in [0.0, 1 / 12, 1 / 6]:
        expected, _ = quad_vec(
            lambda t: expm(model.A * t) @ rate @ expm(model.A.T * t),
            0.0, dt, epsabs=1e-25, epsrel=1e-12,
        )
        *_, factor = model.discretization(dt)
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        np.testing.assert_allclose(
            (factor.T @ factor - expected) / np.where(scale, scale, 1.0),
            0.0, atol=1e-9,
        )


def test_diffusion_long_stable_interval():
    """The inverse-dynamics block must not overflow for a stable physical model."""
    covariance = discretization.diffusion_mfd(-np.eye(2), np.eye(2), 1000.0)
    np.testing.assert_allclose(covariance, 0.5 * np.eye(2), atol=1e-13)


@pytest.mark.parametrize("product", [False, True])
def test_composed_gp_diffusion_against_quadrature(product):
    from scipy.integrate import quad_vec
    from scipy.linalg import expm

    from pysip.statespace import Matern32, Matern52

    left = Matern52([
        {"name": "mscale", "value": 2112.742111978681},
        {"name": "lscale", "value": 481.8830148206853},
        {"name": "sigv", "value": 0.2},
    ], name="trend")
    right = Matern32([
        {"name": "mscale", "value": 1.0},
        {"name": "lscale", "value": 200.0},
        {"name": "sigv", "value": 0.0},
    ], name="modulator")
    model = left * right if product else left + right
    model.update()
    first, second = model._gp1, model._gp2
    if product:
        rate = (
            np.kron(first.Q.T @ first.Q, second.P0.T @ second.P0)
            + np.kron(first.P0.T @ first.P0, second.Q.T @ second.Q)
        )
        np.testing.assert_allclose(
            model.Q.T @ model.Q, rate, rtol=1e-12, atol=1e-14 * np.max(np.abs(rate))
        )
    else:
        rate = model.Q.T @ model.Q
    for dt in [1 / 12, 0.17]:
        expected, _ = quad_vec(
            lambda t: expm(model.A * t) @ rate @ expm(model.A.T * t),
            0.0, dt, epsabs=1e-25, epsrel=1e-12,
        )
        *_, factor = model.discretization(dt)
        scale = np.sqrt(np.outer(np.diag(expected), np.diag(expected)))
        np.testing.assert_allclose(
            (factor.T @ factor - expected) / np.where(scale, scale, 1.0),
            0.0, atol=1e-9,
        )


def test_composed_gp_likelihood_irregular_missing_observations():
    """Discrete composition retains the dense kernel model with missing rows."""
    import pandas as pd

    from pysip.statespace import Matern32, Matern52
    from pysip.statespace_estimator import KalmanQR

    trend = Matern52([
        {"name": "mscale", "value": 8.0},
        {"name": "lscale", "value": 12.0},
        {"name": "sigv", "value": 0.2},
    ], name="trend")
    short = Matern32([
        {"name": "mscale", "value": 1.0},
        {"name": "lscale", "value": 0.7},
        {"name": "sigv", "value": 0.0},
    ], name="short")
    model = trend + trend * short
    times = np.array([0.0, 0.03, 0.12, 0.2, 0.35, 0.7, 0.72, 1.2])
    values = np.array([0.4, np.nan, -0.2, 0.1, np.nan, 0.5, 0.3, -0.1])
    frame = pd.DataFrame({"y": values}, index=times)
    data = model.prepare_data(frame, outputs="y")
    actual = KalmanQR(model).log_likelihood(*data)
    observed = ~np.isnan(values)
    distance = np.abs(times[observed, None] - times[None, observed])
    r = np.sqrt(5.0) * distance / 12.0
    k_trend = 8.0**2 * (1.0 + r + r**2 / 3.0) * np.exp(-r)
    r_short = np.sqrt(3.0) * distance / 0.7
    k_short = (1.0 + r_short) * np.exp(-r_short)
    covariance = k_trend * (1.0 + k_short) + 0.2**2 * np.eye(observed.sum())
    chol = np.linalg.cholesky(covariance)
    residual = np.linalg.solve(chol, values[observed])
    expected = (
        observed.sum() / 2 * np.log(2 * np.pi)
        + np.log(np.diag(chol)).sum() + residual @ residual / 2
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_periodic_diffusion_stays_exactly_zero():
    from pysip.statespace import Periodic

    model = Periodic([
        {"name": "period", "value": 1.0},
        {"name": "mscale", "value": 2.8},
        {"name": "lscale", "value": 1.46},
        {"name": "sigv", "value": 0.0},
    ])
    model.update()
    for dt in [0.0, 1 / 12, 1000.0]:
        *_, factor = model.discretization(dt)
        np.testing.assert_array_equal(factor, np.zeros_like(factor))
