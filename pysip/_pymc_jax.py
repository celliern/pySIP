"""Lazy JAX translation of the PyMC standardized-constrained likelihood."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np

from ._jax import make_likelihood


@contextmanager
def sampling_context() -> Any:
    """Keep PyMC construction, JAX compilation and postprocessing CPU/float64."""
    import jax
    from pytensor.configdefaults import config

    # PyTensor's dispatcher sets global x64 on first import. Undo that side
    # effect before establishing scoped settings, including for native callers.
    previous_x64 = jax.config.jax_enable_x64
    from pytensor.link.jax import dispatch  # noqa: F401

    jax.config.update("jax_enable_x64", previous_x64)
    with (
        config.change_flags(floatX="float64"),
        jax.enable_x64(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        yield


def make_loglike(estimator: Any, data: Any) -> Any:
    """Bind the shared scan without importing optimization priors or transforms.

    Fixed leaves and configuration are captured together with the physical model
    snapshot. PyMC, not this adapter, owns RV transforms and their Jacobians.
    """
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        likelihood = make_likelihood(estimator.ss, data)
    leaves = estimator.ss.parameters.parameters
    coordinates = tuple((p.free, p.loc, p.scale, p.theta) for p in leaves)

    def loglike(standardized: Any) -> Any:
        if standardized.dtype != np.dtype("float64"):
            raise ValueError(
                "PyMC JAX likelihood requires float64: wrap compilation and sampling "
                "in jax.enable_x64() and jax.default_device(jax.devices('cpu')[0])."
            )
        physical = []
        index = 0
        for free, loc, scale, fixed in coordinates:
            if free:
                physical.append(loc + scale * standardized[index])
                index += 1
            else:
                physical.append(jnp.asarray(fixed, dtype=standardized.dtype))
        theta = jnp.stack(physical) if physical else jnp.empty((0,), dtype=standardized.dtype)
        value, bad_row = likelihood(theta)
        # Invalid proposals have zero density, never repaired covariances/jitter.
        return jnp.where((bad_row < 0) & jnp.isfinite(value), value, jnp.inf)

    return loglike
