"""Periodic kernels modulated by Matérn temporal decay."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import GPModel
from ._bessel import periodic_variances


@dataclass
class QuasiPeriodic12(GPModel):
    """Quasi Periodic covariance function, e.g. Periodic x Matern12

    `iv` is the modified Bessel function of the first kind.
    Useful relation: iv(J+1, x) = iv(J-1, x) - 2*J/x * iv(J, x)

    Parameters
    ----------
    J : int, default 7
        Degree of approximation.

    References
    ----------
    Arno Solin and Simo Särkkä (2014). Explicit link between periodic
    covariance functions and state space models. In Proceedings of the
    Seventeenth International Conference on Artifcial Intelligence and
    Statistics (AISTATS 2014). JMLR: W&CP, volume 33.
    """

    J: int = field(default=7)

    # The state-space is composed of J+1 states_block
    states_block = (
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
    )

    params = (
        ("PERIOD", "period", "period of the function"),
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
        ("LENGTH_SCALE", "decay", "control the decay of the periodicity"),
    )

    inputs = ()

    outputs = (("ANY", "sum(f(t))", "sum of stochastic processes"),)

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self.C[0, ::2] = 1.0
        self._kron = np.kron(
            np.diag(range(self.J + 1)),
            np.array([[0.0, -2.0 * np.pi], [2.0 * np.pi, 0.0]]),
        )

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Return decaying harmonic dynamics and full covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Decaying harmonics, configured observations/mean, and covariances.
        """
        period, mscale, lscale, sigv, decay, *_ = theta
        p0 = xp.diag(xp.repeat(periodic_variances(mscale, lscale, self.J, xp=xp), 2))
        return ContinuousMatrices(
            xp.asarray(self._kron) / period - xp.eye(self.nx) / decay,
            xp.zeros((self.nx, 0)), xp.asarray(self.C), xp.asarray(self.D),
            2.0 * p0 / decay, xp.asarray([[sigv**2]]), xp.asarray(self.x0), p0,
        )


@dataclass
class QuasiPeriodic32(GPModel):
    """Quasi Periodic covariance function, e.g. Periodic x Matern32

    `iv` is the modified Bessel function of the first kind.
    Useful relation: iv(J+1, x) = iv(J-1, x) - 2*J/x * iv(J, x)

    Parameters
    ----------
    J : int, default 7
        Degree of the periodic covariance approximation.

    References
    ----------
    Arno Solin and Simo Särkkä (2014). Explicit link between periodic
    covariance functions and state space models. In Proceedings of the
    Seventeenth International Conference on Artificial Intelligence and
    Statistics (AISTATS 2014). JMLR: W&CP, volume 33.

    """

    J: int = field(default=7)

    # The state-space is composed of J+1 states_block
    states_block = (
        ("ANY", "f(t) x f(t)", ""),
        ("ANY", "f(t) x df(t)/dt", ""),
        ("ANY", "df(t) x f(t)", ""),
        ("ANY", "df(t)/dt x df(t)/dt", ""),
    )

    params = (
        ("PERIOD", "period", "period of the function"),
        ("MAGNITUDE_SCALE", "mscale", "control the overall variance of the function"),
        ("LENGTH_SCALE", "lscale", "control the smoothness of the function"),
        ("MEASURE_DEVIATION", "sigv", "measurement standard deviation"),
        ("LENGTH_SCALE", "decay", "control the decay of the periodicity"),
    )

    inputs = ()

    outputs = (("ANY", "sum(f(t))", "sum of stochastic processes"),)

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self.C[0, ::4] = 1.0
        two_pi = 2.0 * np.pi
        self._kron = np.kron(
            np.diag(range(self.J + 1)),
            np.array(
                [
                    [0.0, 0.0, -two_pi, 0.0],
                    [0.0, 0.0, 0.0, -two_pi],
                    [two_pi, 0.0, 0.0, 0.0],
                    [0.0, two_pi, 0.0, 0.0],
                ]
            ),
        )

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Return Matérn-modulated harmonic dynamics and full covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Modulated harmonics, configured observations/mean, and covariances.
        """
        period, mscale, lscale, sigv, decay, *_ = theta
        tmp1 = -3.0 / decay**2
        tmp2 = -2.0 * 3.0**0.5 / decay
        a = xp.asarray(self._kron) / period + xp.kron(
            xp.eye(self.J + 1),
            xp.asarray([
                [0.0, 1.0, 0.0, 0.0],
                [tmp1, tmp2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, tmp1, tmp2],
            ]),
        )
        spectral = xp.diag(xp.repeat(periodic_variances(mscale, lscale, self.J, xp=xp), 2))
        return ContinuousMatrices(
            a, xp.zeros((self.nx, 0)), xp.asarray(self.C), xp.asarray(self.D),
            xp.kron(spectral, xp.diag(xp.asarray([0.0, 4.0 * 3.0**1.5 / decay**3]))),
            xp.asarray([[sigv**2]]), xp.asarray(self.x0),
            xp.kron(spectral, xp.diag(xp.asarray([1.0, 3.0 / decay**2]))),
        )
