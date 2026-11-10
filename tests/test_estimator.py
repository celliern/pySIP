"""Deterministic covariance-form references for the square-root estimator."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pysip.statespace_estimator import (
    KalmanQR,
    States,
    _filtering,
    _log_likelihood,
    _smoothing,
    output_moments,
)


def system(n=4):
    A = np.array([[0.8, 0.2], [-0.1, 0.7]])
    B0 = np.array([[0.4], [-0.2]])
    B1 = np.array([[0.1], [0.3]])
    C = np.array([[1.0, 0.5], [-0.4, 1.2], [0.7, -0.3]])
    D = np.array([[0.3], [-0.6], [0.2]])
    Q = np.array([[0.3, 0.15], [0.0, 0.2]])
    R = np.array([[0.4, 0.2, -0.1], [0.0, 0.5, 0.3], [0.0, 0.0, 0.6]])
    repeat = lambda a: np.repeat(a[..., None], n, axis=-1)
    states = States(repeat(A), repeat(B0), repeat(B1), C, D, repeat(Q), R)
    x0 = np.array([[0.5], [-0.2]])
    P0 = np.array([[0.8, -0.3], [0.0, 0.5]])
    u = np.linspace(-0.4, 0.8, n)[:, None]
    dtu = np.full((n, 1), 0.15)
    return states, x0, P0, u, dtu


def covariance_reference(states, x0, P0, u, dtu, y):
    x, P = x0.copy(), P0.T @ P0
    xs, covs, predicted_x, predicted_P = [], [], [], []
    nll = 0.0
    for i in range(len(y)):
        predicted_x.append(x.copy())
        predicted_P.append(P.copy())
        observed = np.flatnonzero(~np.isnan(y[i]))
        if observed.size:
            C, D = states.C[observed], states.D[observed]
            R = (states.R.T @ states.R)[np.ix_(observed, observed)]
            innovation = y[i, observed, None] - C @ x - D @ u[i, :, None]
            V = C @ P @ C.T + R
            K = np.linalg.solve(V, C @ P).T
            x = x + K @ innovation
            P = P - K @ V @ K.T
            nll += 0.5 * (
                observed.size * np.log(2 * np.pi)
                + np.linalg.slogdet(V)[1]
                + (innovation.T @ np.linalg.solve(V, innovation)).item()
            )
        xs.append(x.copy())
        covs.append(P.copy())
        A, Q = states.A[..., i], states.Q[..., i]
        x = A @ x + states.B0[..., i] @ u[i, :, None] + states.B1[..., i] @ dtu[i, :, None]
        P = A @ P @ A.T + Q.T @ Q
    return np.array(xs), np.array(covs), np.array(predicted_x), np.array(predicted_P), nll


@pytest.mark.parametrize("missing", [False, True])
def test_correlated_filter_smoother_and_likelihood_match_covariance_reference(missing):
    states, x0, P0, u, dtu = system()
    y = np.array([[0.7, -0.3, 0.2], [0.2, 0.5, -0.1], [0.9, 0.1, 0.6], [0.4, -0.2, 0.3]])
    if missing:
        y[0, 1] = np.nan
        y[1, [0, 2]] = np.nan
        y[2] = np.nan
    xs, covs, xp, Pp, nll = covariance_reference(states, x0, P0, u, dtu, y)
    result = _filtering(x0, P0, u, dtu, y, states)
    assert_allclose(result.x, xs, atol=1e-12)
    assert_allclose(result.P, covs, atol=1e-12)
    assert_allclose(_log_likelihood(x0, P0, u, dtu, y, states), nll, atol=1e-12)
    assert np.array_equal(np.isnan(result.k[..., 0]), np.isnan(y))
    for i in range(len(y)):
        observed = np.flatnonzero(~np.isnan(y[i]))
        if observed.size:
            factor = result.S[i][np.ix_(observed, observed)]
            C = states.C[observed]
            expected = C @ Pp[i] @ C.T + (states.R.T @ states.R)[np.ix_(observed, observed)]
            assert_allclose(factor.T @ factor, expected, atol=1e-12)
    for i in range(len(y) - 2, -1, -1):
        gain = np.linalg.solve(Pp[i + 1], states.A[..., i] @ covs[i]).T
        xs[i] += gain @ (xs[i + 1] - xp[i + 1])
        covs[i] += gain @ (covs[i + 1] - Pp[i + 1]) @ gain.T
    smooth = _smoothing(x0, P0, u, dtu, y, states)
    assert_allclose(smooth.x, xs, atol=1e-12)
    assert_allclose(smooth.P, covs, atol=1e-12)


def public_estimator(states, x0, P0):
    return KalmanQR(SimpleNamespace(
        nx=2, ny=3, nu=1, C=states.C, D=states.D, R=states.R,
        x0=x0, P0=P0, update=lambda: None,
        discretization=lambda dt: (
            states.A[..., 0], states.B0[..., 0], states.B1[..., 0], states.Q[..., 0]
        ),
    ))


def test_seeded_simulation_matches_exact_noise_draws_and_same_time_states():
    states, x0, P0, u, dtu = system()
    estimator = public_estimator(states, x0, P0)
    args = (pd.Series(np.ones(len(u))), pd.DataFrame(u), pd.DataFrame(dtu))
    result = estimator.simulate(*args, random_seed=374)
    repeated = estimator.simulate(*args, random_seed=374)
    assert_allclose(repeated.x, result.x, atol=0)
    assert_allclose(repeated.y, result.y, atol=0)
    rng = np.random.RandomState(374)
    x = x0 + P0.T @ rng.randn(2, 1)
    for i in range(len(u)):
        expected_y = states.C @ x + states.D @ u[i, :, None] + states.R.T @ rng.randn(3, 1)
        assert_allclose(result.x[i], x, atol=1e-12)
        assert_allclose(result.y[i], expected_y, atol=1e-12)
        if i + 1 < len(u):
            x = (
                states.A[..., i] @ x + states.B0[..., i] @ u[i, :, None]
                + states.B1[..., i] @ dtu[i, :, None] + states.Q[..., i].T @ rng.randn(2, 1)
            )
    overridden = estimator.simulate(*args, x0=np.zeros_like(x0), P0=np.zeros_like(P0), random_seed=374)
    assert_allclose(overridden.x[0], 0, atol=0)


def test_output_moments_and_public_estimate_use_covariance_variance_sum():
    states, x0, P0, u, dtu = system()
    x = np.repeat(x0[None], len(u), axis=0)
    covariance = np.repeat((P0.T @ P0)[None], len(u), axis=0)
    mean, std = output_moments(states.C, states.D, states.R, x, covariance, u[..., None])
    for i in range(len(u)):
        expected_covariance = states.C @ covariance[i] @ states.C.T + states.R.T @ states.R
        assert_allclose(mean[i], states.C @ x[i] + states.D @ u[i, :, None])
        assert_allclose(std[i, :, 0], np.sqrt(np.diag(expected_covariance)))
    estimator = public_estimator(states, x0, P0)
    y = np.full((len(u), 3), np.nan)
    args = (pd.Series(np.ones(len(u))), pd.DataFrame(u), pd.DataFrame(dtu), pd.DataFrame(y))
    filtered = estimator.filtering(*args)
    estimated = estimator.estimate_output(*args)
    expected_mean, expected_std = output_moments(
        states.C, states.D, states.R, filtered.x, filtered.P, u[..., None]
    )
    assert_allclose(estimated.y, expected_mean)
    assert_allclose(estimated.y_std, expected_std)
    assert estimator.log_likelihood(*args) == 0.0


def test_proxy_normalizes_noncontiguous_dataframe_inputs_once():
    states, x0, P0, _, _ = system()
    estimator = public_estimator(states, x0, P0)
    frame = pd.DataFrame(np.arange(16.0).reshape(4, 4)[:, ::2])
    converted = estimator._proxy_params(pd.Series(np.ones(4)), (frame, frame, frame))
    assert all(value.flags.c_contiguous for value in converted[:-1])
