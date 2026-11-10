"""Scaled periodic spectral coefficients with lazy analytic JAX derivatives."""

from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from scipy.special import ive


def _scaled_bessel(argument: Any, degree: int) -> np.ndarray:
    """Return ``exp(-argument) I_n(argument)`` for nonnegative arguments.

    Parameters
    ----------
    argument : scalar
        Nonnegative inverse squared periodic length scale.
    degree : int
        Highest harmonic order.

    Returns
    -------
    numpy.ndarray
        Scaled modified Bessel values in ascending harmonic order.
    """
    return np.asarray(ive(np.arange(degree + 1), argument), dtype=np.asarray(argument).dtype)


@cache
def _jax_scaled_bessel(degree: int) -> Callable[[Any], Any]:
    """Build a lazy callback with the exact Bessel recurrence derivative.

    Parameters
    ----------
    degree : int
        Highest harmonic order, fixed when tracing.

    Returns
    -------
    callable
        JAX-compatible scaled Bessel evaluator with an analytic JVP.
    """
    import jax
    import jax.numpy as jnp

    def table(argument: Any) -> Any:
        """Transfer IEEE bits so callback-worker x64 settings cannot downcast."""
        dtype = np.dtype(argument.dtype)
        words = dtype.itemsize // np.dtype(np.uint32).itemsize

        def callback(bits: Any) -> np.ndarray:
            """Evaluate SciPy on host arrays, never on JAX callback arguments."""
            value = np.asarray(bits, dtype=np.uint32).reshape(-1).view(dtype).reshape(())
            values = _scaled_bessel(value, degree + 1)
            return values.view(np.uint32).reshape(degree + 2, words)

        packed = jax.pure_callback(
            callback,
            jax.ShapeDtypeStruct((degree + 2, words), jnp.uint32),
            jax.lax.bitcast_convert_type(argument, jnp.uint32),
            vmap_method="sequential",
        )
        if words == 1:
            packed = packed[:, 0]
        return jax.lax.bitcast_convert_type(packed, argument.dtype)

    @jax.custom_jvp
    def evaluate(argument: Any) -> Any:
        """Evaluate the scaled Bessel vector through a precision-safe callback."""
        return table(argument)[:-1]

    @evaluate.defjvp
    def evaluate_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[Any, Any]:
        """Propagate tangents using the exact neighboring-order recurrence."""
        argument, = primals
        tangent, = tangents
        values = table(argument)
        lower = jnp.concatenate((values[1:2], values[:degree]))
        slope = 0.5 * (lower + values[1:]) - values[:-1]
        return values[:-1], slope * tangent

    return evaluate


def periodic_variances(mscale: Any, lscale: Any, degree: int, *, xp: Any = np) -> Any:
    """Return stable finite-harmonic variances without unscaled Bessel overflow.

    Parameters
    ----------
    mscale : scalar
        Process standard deviation.
    lscale : scalar
        Positive periodic length scale.
    degree : int
        Highest retained harmonic order.
    xp : module, optional
        NumPy or JAX NumPy array namespace.

    Returns
    -------
    array
        Variances, with the zero-frequency coefficient weighted once.
    """
    argument = xp.asarray(lscale ** -2)
    values = (_scaled_bessel(argument, degree) if xp is np
              else _jax_scaled_bessel(degree)(argument))
    weights = xp.where(xp.arange(degree + 1) == 0, 1.0, 2.0)
    variances = weights * mscale**2 * values
    if xp is np and not np.all(np.isfinite(variances)):
        raise ValueError("Spectral variance coefficients are not finite!")
    return variances


def periodic_scales(mscale: Any, lscale: Any, degree: int, *, xp: Any = np) -> Any:
    """Return spectral factors with finite tangents at zero overall amplitude.

    Pulling the amplitude outside the square root avoids the undefined
    ``sqrt(mscale**2)`` derivative at zero. Underflowed harmonics remain zero.
    Squared factors equal ``periodic_variances`` without changing covariance.
    """
    weights = periodic_variances(1.0, lscale, degree, xp=xp)
    root = xp.sqrt(xp.where(weights > 0.0, weights, 1.0))
    return xp.abs(mscale) * xp.where(weights > 0.0, root, 0.0)
