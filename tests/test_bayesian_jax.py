"""Real compiled likelihoods and posterior draws, not sampler wiring mocks."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pysip import Regressor
from pysip._pymc_jax import sampling_context
from pysip.params.prior import Gamma, Normal
from pysip.statespace import ContinuousMatrices, Matern32, StateSpace


class LinearGain(StateSpace):
    """Independent Normal observations with a canonical, zero-noise latent state."""

    states = (("ANY", "x", "unused deterministic state"),)
    inputs = (("ANY", "u", "regressor"),)
    outputs = (("ANY", "y", "measurement"),)
    params = (("ANY", "gain", "physical gain"), ("ANY", "noise", "fixed deviation"))

    def continuous_matrices(self, theta, *, xp=np):
        gain, noise = theta
        zero = xp.zeros((1, 1))
        return ContinuousMatrices(
            xp.array([[-1.0]]), zero, zero, xp.reshape(gain, (1, 1)),
            zero, xp.reshape(noise**2, (1, 1)), zero, zero,
        )


def linear_case():
    reg = Regressor(LinearGain([
        {"name": "gain", "value": 0.2, "loc": 1.1, "scale": 2.3,
         "transform": "none", "prior": Normal(-0.1, 0.7)},
        {"name": "noise", "value": 0.5, "loc": 0.2, "scale": 0.8,
         "transform": "fixed"},
    ]))
    u = np.array([-1.2, 0.2, 1.4, -0.7, 0.9, 1.8, -0.4, 0.6])
    data = pd.DataFrame({
        "u": u, "y": 1.7 * u + np.array([0.2, -0.3, 0.1, -0.2, 0.4, -0.1, 0.3, 0.1]),
    }, index=np.arange(len(u), dtype=float) * 0.2)
    data.loc[data.index[[2, 7]], "y"] = np.nan
    return reg, data


def analytic_posterior(reg, data):
    gain, noise = reg.parameters.parameters
    mask = data.y.notna()
    design = gain.scale * data.loc[mask, "u"].to_numpy()
    centered = data.loc[mask, "y"].to_numpy() - gain.loc * data.loc[mask, "u"].to_numpy()
    variance = 1.0 / (1.0 / 0.7**2 + design @ design / noise.theta**2)
    mean = variance * (-0.1 / 0.7**2 + design @ centered / noise.theta**2)
    return mean, variance


def test_compiled_logp_and_both_gradient_paths_match_native_and_conjugacy():
    from pymc.sampling.jax import get_jaxified_graph, get_jaxified_logp

    reg, data = linear_case()
    initial = reg.parameters.theta.copy()
    mean, variance = analytic_posterior(reg, data)
    with sampling_context():
        model = reg.pymc_model(data)
        logp = jax.jit(get_jaxified_logp(model))
        gradient = jax.jit(jax.grad(get_jaxified_logp(model)))
        symbolic_gradient = jax.jit(get_jaxified_graph(model.value_vars, [model.dlogp()]))
        native_logp, native_gradient = model.compile_logp(), model.compile_dlogp()
        for value in [-0.5, 0.1, 0.8]:
            point = {"gain": np.asarray(value)}
            vector = [jnp.asarray(value, dtype=jnp.float64)]
            np.testing.assert_allclose(logp(vector), native_logp(point), atol=1e-10)
            expected = -(value - mean) / variance
            np.testing.assert_allclose(gradient(vector)[0], expected, atol=1e-9)
            np.testing.assert_allclose(symbolic_gradient(*vector)[0], [expected], atol=1e-9)
            np.testing.assert_allclose(native_gradient(point), [expected], rtol=2e-5, atol=2e-5)
        # Built graphs must not read subsequently mutated live parameters.
        reg.parameters.theta = [10.0, 2.0]
        point = {"gain": np.asarray(0.1)}
        np.testing.assert_allclose(logp([jnp.asarray(0.1)]), native_logp(point), atol=1e-10)
    np.testing.assert_allclose(initial, [1.56, 0.6], atol=1e-15)


def test_transformed_pymc_prior_jacobian_is_counted_once():
    from pymc.sampling.jax import get_jaxified_graph, get_jaxified_logp

    reg = Regressor(Matern32([
        {"name": "mscale", "value": 0.7, "loc": 0.2, "scale": 1.8,
         "transform": "log", "prior": Gamma(3.0, 4.0)},
        {"name": "lscale", "value": 0.4, "transform": "fixed"},
        {"name": "sigv", "value": 0.15, "transform": "fixed"},
    ]), outputs="y")
    data = pd.DataFrame({"y": [0.1, np.nan, -0.2, 0.3, np.nan]}, index=[0., .1, .3, .6, .9])
    with sampling_context():
        model = reg.pymc_model(data)
        logp = jax.jit(get_jaxified_logp(model))
        gradient = jax.jit(jax.grad(get_jaxified_logp(model)))
        symbolic = jax.jit(get_jaxified_graph(model.value_vars, [model.dlogp()]))
        native_logp, native_gradient = model.compile_logp(), model.compile_dlogp()
        for value in [-1.0, -0.2, 0.4]:
            point = {"mscale_log__": np.asarray(value)}
            vector = [jnp.asarray(value, dtype=jnp.float64)]
            np.testing.assert_allclose(logp(vector), native_logp(point), atol=1e-9)
            expected = native_gradient(point)
            np.testing.assert_allclose(gradient(vector), expected, rtol=3e-5, atol=3e-5)
            np.testing.assert_allclose(symbolic(*vector)[0], expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("sampler", ["numpyro", "blackjax"])
@pytest.mark.parametrize("direct", [False, True])
def test_real_jax_nuts_recovers_analytic_normal_posterior(sampler, direct):
    pytest.importorskip(sampler)
    reg, data = linear_case()
    initial = reg.parameters.theta.copy()
    mean, variance = analytic_posterior(reg, data)
    options = {
        "draws": 800, "tune": 400, "chains": 2, "random_seed": 1782,
        "progressbar": False, "compute_convergence_checks": False,
        "nuts_sampler": sampler,
        "nuts": {"chain_method": "vectorized", "target_accept": 0.9,
                 "postprocessing_backend": "cpu"},
    }
    if direct:
        with sampling_context(), reg.pymc_model(data):
            trace = pm.sample(**options)
    else:
        trace = reg.sample(data, **options)
    draws = trace["posterior"]["gain"].values.reshape(-1)
    assert np.isfinite(draws).all()
    assert abs(draws.mean() - mean) < 0.15 * np.sqrt(variance)
    np.testing.assert_allclose(draws.var(), variance, rtol=0.2)
    assert not trace["sample_stats"]["diverging"].values.any()
    assert "noise" not in trace["posterior"].data_vars
    np.testing.assert_array_equal(reg.parameters.theta, initial)
