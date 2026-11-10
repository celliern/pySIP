"""Opt-in fitting must preserve the reference statistical model."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pysip import Regressor
from pysip._jax import make_objective
from pysip.params.prior import (
    BasePrior,
    Beta,
    Gamma,
    InverseGamma,
    LogNormal,
    Normal,
    Uniform,
)
from pysip.params.transforms import (
    FixedTransform,
    LogitTransform,
    LogTransform,
    LowerTransform,
    NoneTransform,
    UpperTransform,
)
from pysip.statespace import (
    Matern32,
    Periodic,
    QuasiPeriodic12,
    QuasiPeriodic32,
    StateSpace,
    array_set,
)


class CoupledModel(StateSpace):
    """A custom model with correlated measurements and rank-deficient noise."""

    states = (("ANY", "x1", "first state"), ("ANY", "x2", "second state"))
    inputs = (("ANY", "u", "input"),)
    outputs = (("ANY", "y1", "first output"), ("ANY", "y2", "second output"))
    params = (("ANY", "decay", "decay rate"),)
    zero_covariances = False

    def update_continuous_ssm(self):
        decay = self.parameters.theta[0]
        self.A = array_set(self.A, slice(None), [[-decay, 0.3], [0.0, -0.4]])
        self.B = array_set(self.B, slice(None), [[0.4], [0.2]])
        self.C = array_set(self.C, slice(None), [[1.0, 0.2], [0.1, 1.0]])
        self.D = array_set(self.D, slice(None), [[0.3], [-0.2]])
        self.Q = array_set(self.Q, (1, 1), 0.0 if self.zero_covariances else 0.05)
        self.R = array_set(self.R, slice(None), [[0.2, 0.12], [0.0, 0.3]])
        self.x0 = array_set(self.x0, slice(None), [[0.2], [-0.1]])
        self.P0 = array_set(self.P0, (0, 0), 0.0 if self.zero_covariances else 0.4)


def coupled_case(zero_covariances=False, hold_order=0):
    model = CoupledModel([
        {"name": "decay", "value": 0.4, "loc": 0.1, "scale": 2.0,
         "bounds": (0.0, 2.0), "transform": "logit", "prior": Normal(0.4, 0.3)},
    ], hold_order=hold_order)
    model.zero_covariances = zero_covariances
    data = pd.DataFrame({
        "u": [0.2, 0.8, -0.3, 0.4, 0.1, 0.6],
        "y1": [0.2, np.nan, 0.1, np.nan, 0.3, np.nan],
        "y2": [-0.1, 0.2, np.nan, np.nan, -0.1, np.nan],
    }, index=[0.0, 0.15, 0.4, 0.9, 1.2, 1.5])
    return Regressor(model), data


def reference_objective(reg, data):
    reference = deepcopy(reg)

    def objective(eta):
        reference.parameters.eta_free = eta
        return reference.log_likelihood(data) - reference.parameters.prior + reference.parameters.penalty

    return objective


@pytest.mark.parametrize("zero_covariances,hold_order", [(False, 0), (True, 1)])
def test_custom_multioutput_gradient_preserves_missingness_and_covariance_boundaries(
    zero_covariances, hold_order,
):
    reg, data = coupled_case(zero_covariances, hold_order)
    initial = reg.parameters.theta.copy()
    x64 = jax.config.jax_enable_x64
    objective = make_objective(reg, reg.prepare_data(data))
    reference = reference_objective(reg, data)
    eta = reg.parameters.eta_free + 0.15
    value, gradient = objective(eta)
    expected_gradient = approx_derivative(reference, eta, method="3-point").reshape(-1)
    np.testing.assert_allclose(value, reference(eta), rtol=0, atol=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(reg.parameters.theta, initial)
    assert jax.config.jax_enable_x64 == x64
    # Trailing missing forecasts do not alter the posterior or its derivative.
    shortened = make_objective(reg, reg.prepare_data(data.iloc[:-1]))
    np.testing.assert_allclose(shortened(eta)[0], value, rtol=0, atol=1e-10)
    np.testing.assert_allclose(shortened(eta)[1], gradient, rtol=0, atol=1e-10)


def test_builtin_prior_and_transform_derivatives_match_reference():
    priors = [Normal(0.3, 0.8), Gamma(2.0, 3.0), Beta(2.0, 4.0),
              InverseGamma(3.0, 2.0), LogNormal(0.2, 0.7), Uniform(-0.5, 1.5)]
    transforms = [NoneTransform((None, None)), FixedTransform((None, None)), LogTransform((0.0, None)),
                  LowerTransform((-2.0, None)), UpperTransform((None, 3.0)),
                  LogitTransform((-2.0, 3.0))]
    with jax.enable_x64(True):
        for prior in priors:
            value, gradient = jax.value_and_grad(prior.jax_logpdf)(jnp.asarray(0.4))
            expected = approx_derivative(lambda x, prior=prior: prior.logpdf(x[0]), np.array([0.4])).item()
            np.testing.assert_allclose(value, prior.logpdf(0.4), rtol=1e-12)
            np.testing.assert_allclose(gradient, expected, rtol=1e-7, atol=1e-8)
            np.testing.assert_allclose(
                jax.vmap(prior.jax_logpdf)(jnp.array([-1.0, 0.2, 2.0])),
                prior.logpdf(np.array([-1.0, 0.2, 2.0])), rtol=1e-12,
            )
        for transform in transforms:
            theta = transform.untransform(0.2)
            value, gradient = jax.value_and_grad(lambda eta, transform=transform: transform.untransform(eta, xp=jnp))(jnp.asarray(0.2))
            np.testing.assert_allclose(value, theta, rtol=1e-12)
            np.testing.assert_allclose(gradient, transform.grad_untransform(0.2), rtol=1e-12)
            value, gradient = jax.value_and_grad(lambda x, transform=transform: transform.penalty(x, xp=jnp))(jnp.asarray(theta))
            np.testing.assert_allclose(value, transform.penalty(theta), rtol=1e-12, atol=1e-15)
            np.testing.assert_allclose(gradient, transform.grad_penalty(theta), rtol=1e-12, atol=1e-15)


def test_jax_fit_preserves_fixed_parameters_and_fitted_uncertainty():
    reg = Regressor(Matern32([
        {"name": "mscale", "value": 1.0, "transform": "fixed"},
        {"name": "lscale", "value": 0.8, "scale": 0.5, "transform": "log", "prior": Gamma(3.0, 2.0)},
        {"name": "sigv", "value": 0.1, "transform": "fixed"},
    ]), outputs="y")
    reference = deepcopy(reg)
    time = np.linspace(0.0, 1.0, 12)
    data = pd.DataFrame({"y": np.sin(12.0 * time)}, index=time)
    summary, _, result = reg.fit(data, backend="jax", disp=False)
    expected_summary, _, expected_result = reference.fit(data, disp=False)
    assert result.success and expected_result.success
    np.testing.assert_allclose(result.fun, expected_result.fun, rtol=0, atol=1e-8)
    np.testing.assert_allclose(summary[["θ", "σ(θ)"]], expected_summary[["θ", "σ(θ)"]], rtol=1e-3)
    np.testing.assert_array_equal(reg.parameters.theta[[0, 2]], [1.0, 0.1])


def test_unsupported_customizations_and_singular_innovations_fail_explicitly():
    class LegacyModel(CoupledModel):
        def update_continuous_ssm(self):
            super().update_continuous_ssm()
            self.A[0, 0] = -self.parameters.theta[0]

    reg, data = coupled_case()
    legacy = Regressor(LegacyModel(deepcopy(reg.parameters)))
    np.testing.assert_allclose(legacy.log_likelihood(data), reg.log_likelihood(data), atol=1e-12)
    with pytest.raises(TypeError, match="immutable|continuous_matrices"):
        legacy.fit(data, backend="jax", disp=False)

    class SingularModel(CoupledModel):
        def update_continuous_ssm(self):
            super().update_continuous_ssm()
            self.R = array_set(self.R, slice(None), [[0.0, 0.0], [0.0, 0.0]])

    singular = SingularModel(deepcopy(reg.parameters))
    singular.zero_covariances = True
    with pytest.raises(ValueError, match="finite|positive|innovation"):
        Regressor(singular).fit(data, backend="jax", disp=False)

    class CustomPrior(BasePrior):
        @property
        def scipy_dist(self):
            return Normal().scipy_dist

    with pytest.raises(NotImplementedError, match="jax_logpdf|custom"):
        CustomPrior().jax_logpdf(jnp.asarray(0.4))


@pytest.mark.parametrize("model_type", [Periodic, QuasiPeriodic12, QuasiPeriodic32])
def test_zero_periodic_amplitude_has_finite_likelihood_derivatives(model_type):
    parameters = [
        {"name": "period", "value": 1.0, "transform": "fixed"},
        {"name": "mscale", "value": 0.0, "transform": "none"},
        {"name": "lscale", "value": 0.7, "transform": "log"},
        {"name": "sigv", "value": 0.2, "transform": "fixed"},
    ]
    if model_type is not Periodic:
        parameters.append({"name": "decay", "value": 1.2, "transform": "fixed"})
    reg = Regressor(model_type(parameters, J=1), outputs="y")
    data = pd.DataFrame({"y": [0.1, -0.1, 0.2]}, index=[0.0, 0.2, 0.5])
    objective = make_objective(reg, reg.prepare_data(data))
    value, gradient = objective(reg.parameters.eta_free)
    reference = reference_objective(reg, data)
    np.testing.assert_allclose(value, reference(reg.parameters.eta_free), atol=1e-10)
    np.testing.assert_allclose(gradient, 0.0, atol=1e-10)


def test_custom_discretization_is_preserved_or_rejected_including_nested_models():
    class LegacyKernel(Matern32):
        @staticmethod
        def changed_transition(parts):
            a, b0, b1, q = parts
            return 0.9 * a, b0, b1, q

        def discretization(self, dt):
            return self.changed_transition(super().discretization(dt))

    class PureKernel(LegacyKernel):
        def discrete_covariance(self, theta, dt, *, xp=np):
            return self.changed_transition(super().discrete_covariance(theta, dt, xp=xp))

    specs = [
        {"name": "mscale", "value": 0.8, "transform": "fixed"},
        {"name": "lscale", "value": 0.6, "transform": "log"},
        {"name": "sigv", "value": 0.2, "transform": "fixed"},
    ]
    kernel = LegacyKernel(specs)
    data = pd.DataFrame({"y": [0.1, -0.2, 0.4, 0.2]}, index=[0.0, 0.2, 0.4, 0.7])
    for model in (kernel, Matern32(deepcopy(kernel.parameters)) + kernel):
        reg = Regressor(model, outputs="y")
        with pytest.raises(NotImplementedError, match="implement discrete_covariance"):
            reg.fit(data, backend="jax", disp=False)
    reg = Regressor(PureKernel(specs), outputs="y")
    value, gradient = make_objective(reg, reg.prepare_data(data))(reg.parameters.eta_free)
    reference = reference_objective(reg, data)
    np.testing.assert_allclose(value, reference(reg.parameters.eta_free), atol=1e-10)
    np.testing.assert_allclose(
        gradient, approx_derivative(reference, reg.parameters.eta_free).reshape(-1),
        rtol=1e-5, atol=1e-7,
    )
