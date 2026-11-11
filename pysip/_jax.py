"""Optional CPU/x64 automatic differentiation of the covariance Kalman likelihood."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .params.parameters import Parameters
from .regressors import Regressor
from .statespace.base import GPModel, RCModel, StateSpace


def make_likelihood(
    ss: StateSpace,
    data: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Callable[[Any], tuple[Any, Any]]:
    """Snapshot a model/data pair and return pure JAX NLL in physical coordinates.

    The second result is the first invalid innovation row, or -1. Both Bayesian
    lowering and frequentist fitting consume this covariance scan; no priors,
    coordinate transforms, penalties or Jacobians belong here. JAX is imported
    only on demand. Construction and execution require CPU/float64 contexts.
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax.scipy.linalg import cho_solve, solve_triangular
    except ImportError as error:
        raise ImportError(
            "JAX fitting requires the optional extra: pip install 'pysip[jax]'"
        ) from error

    model = deepcopy(ss)

    def validate_discretization(instance: StateSpace) -> None:
        """Never ignore a custom legacy discretization, including nested models."""
        hierarchy = type(instance).__mro__
        legacy = next(cls for cls in hierarchy if "discretization" in vars(cls))
        pure = next(cls for cls in hierarchy if "discrete_covariance" in vars(cls))
        if legacy not in (StateSpace, RCModel, GPModel) and hierarchy.index(legacy) < hierarchy.index(pure):
            raise NotImplementedError(
                f"{type(instance).__name__} overrides discretization; "
                "implement discrete_covariance to preserve it during JAX fitting"
            )
        for value in vars(instance).values():
            if isinstance(value, StateSpace):
                validate_discretization(value)

    validate_discretization(model)
    dt, u, dtu, y = data
    observations = np.asarray(y, dtype=np.float64)
    observed = ~np.isnan(observations)
    rows = np.flatnonzero(observed.any(axis=1))
    n = int(rows[-1] + 1) if rows.size else 0
    # The final transition is never consumed, including for a single-row scan.
    times, indices = np.unique(
        np.asarray(dt, dtype=np.float64)[: max(n - 1, 1)], return_inverse=True,
    )
    selected = np.zeros(n, dtype=np.int64)
    selected[: max(n - 1, 0)] = indices[: max(n - 1, 0)]
    # Keep captured data on the host: this factory can run during an outer
    # PyMC trace, where caching jnp.asarray results would leak JAX tracers.
    inputs = np.array(u, dtype=np.float64, copy=True)[:n]
    derivatives = np.array(dtu, dtype=np.float64, copy=True)[:n]
    outputs = np.where(observed[:n], observations[:n], 0.0)
    masks = observed[:n]
    time_values = times
    time_indices = selected
    row_indices = np.arange(n)

    def likelihood(theta: Any) -> tuple[Any, Any]:
        """Evaluate negative likelihood and identify nonpositive innovations."""
        if not n:
            return jnp.asarray(0.0, dtype=theta.dtype), jnp.asarray(-1)

        matrices = model.continuous_matrices(theta, xp=jnp)
        initial_covariance = jnp.asarray(matrices.P0c)
        diagonal = jnp.diag(initial_covariance)
        # Select before sqrt: sqrt(0)'s undefined tangent must never be formed.
        scale = jnp.sqrt(jnp.where(diagonal > 0.0, diagonal, 1.0))
        outer_scale = scale[:, None] * scale[None, :]
        x0 = jnp.asarray(matrices.x0).reshape(-1) / scale
        p0 = initial_covariance / outer_scale
        c = jnp.asarray(matrices.C) * scale[None, :]
        d = jnp.asarray(matrices.D)
        r = jnp.asarray(matrices.Rc)
        identity = jnp.eye(x0.shape[0], dtype=x0.dtype)
        a, b0, b1, q = jax.vmap(
            lambda interval: model.discrete_covariance(theta, interval, xp=jnp),
        )(time_values)
        a = a * scale[None, None, :] / scale[None, :, None]
        b0 = b0 / scale[None, :, None]
        b1 = b1 / scale[None, :, None]
        q = q / outer_scale[None, :, :]

        def step(carry: Any, item: Any) -> tuple[Any, None]:
            """Marginalize missing channels, Joseph-update, then predict in time."""
            x, p, total, bad_row = carry
            index, ui, dui, yi, mask, row = item
            weight = mask.astype(x.dtype)
            h = c * weight[:, None]
            noise = r * weight[:, None] * weight[None, :] + jnp.diag(1.0 - weight)
            pc = p @ h.T
            innovation = h @ pc + noise
            factor = jnp.linalg.cholesky(innovation)
            valid = jnp.all(jnp.isfinite(factor)) & jnp.all(jnp.diag(factor) > 0.0)
            bad_row = jnp.where((bad_row < 0) & ~valid, row, bad_row)
            residual = jnp.where(mask, yi - c @ x - d @ ui, 0.0)
            whitened = solve_triangular(factor, residual, lower=True)
            gain = cho_solve((factor, True), pc.T).T
            left = identity - gain @ h
            updated_p = left @ p @ left.T + gain @ noise @ gain.T
            x = x + gain @ residual
            p = (updated_p + updated_p.T) * 0.5
            total = total + 0.5 * (
                jnp.sum(weight) * jnp.log(2.0 * jnp.pi)
                + 2.0 * jnp.sum(jnp.log(jnp.diag(factor)))
                + whitened @ whitened
            )

            def predict(state: Any) -> tuple[Any, Any]:
                """Propagate normalized state mean and full covariance."""
                mean, covariance = state
                transition = a[index]
                predicted = transition @ covariance @ transition.T + q[index]
                return (
                    transition @ mean + b0[index] @ ui + b1[index] @ dui,
                    (predicted + predicted.T) * 0.5,
                )

            x, p = jax.lax.cond(row + 1 < n, predict, lambda state: state, (x, p))
            return (x, p, total, bad_row), None

        (_, _, value, bad_row), _ = jax.lax.scan(
            step,
            (x0, p0, jnp.asarray(0.0, dtype=theta.dtype), jnp.asarray(-1)),
            (time_indices, inputs, derivatives, outputs, masks, row_indices),
        )
        return value, bad_row

    return likelihood


def make_objective(
    reg: Regressor,
    data: tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]:
    """Build a validated CPU/float64 posterior value/gradient in free eta.

    Unlike the shared physical likelihood, this optimization adapter includes
    frequentist parameter transforms, priors and penalties, without Jacobians.
    """
    likelihood = make_likelihood(reg.ss, data)
    import jax
    import jax.numpy as jnp

    enable_x64 = jax.enable_x64
    cpu = jax.devices("cpu")[0]
    parameters = cast("Parameters", deepcopy(reg.parameters))
    leaves = parameters.parameters
    eta0 = np.asarray(parameters.eta_free, dtype=np.float64).copy()

    def objective(eta: Any) -> tuple[Any, Any]:
        physical = []
        offset = jnp.asarray(0.0)
        free_index = 0
        for parameter in leaves:
            if parameter.free:
                standardized = parameter._transform.untransform(
                    eta[free_index], xp=jnp,
                )
                free_index += 1
                offset = offset + 1e-4 * parameter._transform.penalty(
                    standardized, xp=jnp,
                )
                if parameter.prior is not None:
                    offset = offset - parameter.prior.jax_logpdf(standardized)
            else:
                standardized = jnp.asarray(parameter.theta_sd)
            physical.append(parameter.loc + parameter.scale * standardized)
        theta = jnp.stack(physical) if physical else jnp.empty((0,))
        value, bad_row = likelihood(theta)
        return value + offset, bad_row

    with enable_x64(), jax.default_device(cpu):
        compiled = jax.jit(jax.value_and_grad(objective, has_aux=True))

    def evaluate(eta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        """Evaluate on CPU/x64 and materialize SciPy-compatible host results."""
        with enable_x64(), jax.default_device(cpu):
            (value, bad_row), gradient = compiled(jnp.asarray(eta, dtype=jnp.float64))
            result = float(value)
            bad_index = int(bad_row)
            host_gradient = np.asarray(gradient, dtype=np.float64)
        if bad_index >= 0:
            raise ValueError(
                f"JAX innovation covariance at prepared data row {bad_index} is not "
                "finite positive definite. Check model covariances and observations; "
                "zero process/initial covariance is valid only with positive-definite "
                "observed innovations."
            )
        if not np.isfinite(result):
            raise ValueError("JAX posterior objective is nonfinite at the supplied eta.")
        if not np.isfinite(host_gradient).all():
            raise ValueError("JAX posterior gradient is nonfinite at the supplied eta.")
        return result, host_gradient

    try:
        evaluate(eta0)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.TracerBoolConversionError,
        jax.errors.ConcretizationTypeError,
        jax.errors.UnexpectedTracerError,
    ) as error:
        raise TypeError(
            f"{type(reg.ss).__name__} cannot be differentiated by the JAX backend. "
            "Custom models must provide traceable continuous_matrices(theta, xp=...) "
            "and discrete_covariance(theta, dt, xp=...) operations, or use the "
            "default SciPy fitting backend."
        ) from error
    return evaluate
