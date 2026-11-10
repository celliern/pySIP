"""Matérn kernels with magnitude expressed as a standard deviation."""

from dataclasses import dataclass
from typing import cast

from ...params import Parameters
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
        """Mutate parameter-independent continuous matrices in place."""
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        mscale, lscale, sigv, *_ = cast(Parameters, self.parameters).theta

        self.A[:] = -1.0 / lscale
        self.Q[:] = 2.0**0.5 * mscale / lscale**0.5
        self.R[:] = sigv
        self.P0[:] = mscale


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
        """Mutate parameter-independent continuous matrices in place."""
        self.A[0, 1] = 1.0
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        mscale, lscale, sigv, *_ = cast(Parameters, self.parameters).theta

        self.A[1, :] = [-3.0 / lscale**2, -2.0 * 3.0**0.5 / lscale]
        self.Q[1, 1] = 2.0 * 3.0**0.75 * mscale / lscale**1.5
        self.R[0, 0] = sigv
        self.P0[self._diag] = [mscale, 3.0**0.5 * mscale / lscale]


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
        """Mutate parameter-independent continuous matrices in place."""
        self.A[0, 1] = 1.0
        self.A[1, 2] = 1.0
        self.C[0, 0] = 1.0

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        mscale, lscale, sigv, *_ = cast(Parameters, self.parameters).theta

        self.A[2, :] = [
            -(5.0**1.5) / lscale**3,
            -15.0 / lscale**2,
            -3.0 * 5.0**0.5 / lscale,
        ]
        self.Q[2, 2] = 20.0 * 5.0**0.25 / 3.0**0.5 * mscale / lscale**2.5
        self.R[0, 0] = sigv
        # Upper triangular Cholesky factor
        self.P0[:] = [
            [mscale, 0.0, -5.0 * mscale / (3.0 * lscale**2)],
            [0, 15**0.5 * mscale / (3.0 * lscale), 0.0],
            [0.0, 0.0, 10 * 2**0.5 * mscale / (3.0 * lscale**2)],
        ]
