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
