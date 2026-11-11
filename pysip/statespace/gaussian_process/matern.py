"""Matérn kernels with magnitude expressed as a standard deviation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import GPModel


@dataclass
class Matern12(GPModel):
    """Matérn covariance function with smoothness parameter = 1/2"""

    states = (("ANY", "f(t)", "stochastic process"),)

    params = (
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
    )

    inputs = ()

    outputs = (("ANY", "f(t)", "stochastic process"),)

    def set_constant_continuous_ssm(self) -> None:
        """Initialize the fixed observation matrix."""
        self.C[0, 0] = 1.0

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Return the continuous realization and full covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Dynamics, configured observations/mean, and full covariances.
        """
        mscale, lscale, sigv, *_ = theta
        variance = mscale**2
        return ContinuousMatrices(
            xp.asarray([[-1.0 / lscale]]), xp.zeros((1, 0)),
            xp.asarray(self.C), xp.asarray(self.D),
            xp.asarray([[2.0 * variance / lscale]]), xp.asarray([[sigv**2]]),
            xp.asarray(self.x0), xp.asarray([[variance]]),
        )


@dataclass
class Matern32(GPModel):
    """Matérn covariance function with smoothness parameter = 3/2"""

    states = (
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
    )

    params = (
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
    )

    inputs = ()

    outputs = (("ANY", "f(t)", "stochastic process"),)

    def set_constant_continuous_ssm(self) -> None:
        """Initialize the fixed observation matrix."""
        self.C[0, 0] = 1.0

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Return the continuous realization and full covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Dynamics, configured observations/mean, and full covariances.
        """
        mscale, lscale, sigv, *_ = theta
        variance = mscale**2
        return ContinuousMatrices(
            xp.asarray([[0.0, 1.0], [-3.0 / lscale**2, -2.0 * 3.0**0.5 / lscale]]),
            xp.zeros((2, 0)), xp.asarray(self.C), xp.asarray(self.D),
            xp.diag(xp.asarray([0.0, 4.0 * 3.0**1.5 * variance / lscale**3])),
            xp.asarray([[sigv**2]]), xp.asarray(self.x0),
            xp.diag(xp.asarray([variance, 3.0 * variance / lscale**2])),
        )


@dataclass
class Matern52(GPModel):
    """Matérn covariance function with smoothness parameter = 5/2"""

    states = (
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
        ("ANY", "d²f(t)/d²t", "second derivative stochastic process"),
    )

    params = (
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
    )

    inputs = ()

    outputs = (("ANY", "f(t)", "stochastic process"),)

    def set_constant_continuous_ssm(self) -> None:
        """Initialize the fixed observation matrix."""
        self.C[0, 0] = 1.0

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Return the continuous realization and full covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Dynamics, configured observations/mean, and full covariances.
        """
        mscale, lscale, sigv, *_ = theta
        variance = mscale**2
        a = xp.asarray([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-5.0**1.5 / lscale**3, -15.0 / lscale**2, -3.0 * 5.0**0.5 / lscale],
        ])
        p0 = variance * xp.asarray([
            [1.0, 0.0, -5.0 / (3.0 * lscale**2)],
            [0.0, 5.0 / (3.0 * lscale**2), 0.0],
            [-5.0 / (3.0 * lscale**2), 0.0, 25.0 / lscale**4],
        ])
        return ContinuousMatrices(
            a, xp.zeros((3, 0)), xp.asarray(self.C), xp.asarray(self.D),
            xp.diag(xp.asarray([0.0, 0.0, 400.0 * 5.0**0.5 * variance / (3.0 * lscale**5)])),
            xp.asarray([[sigv**2]]), xp.asarray(self.x0), p0,
        )
