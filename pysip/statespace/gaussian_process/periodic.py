"""Finite harmonic approximation of the periodic covariance kernel."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import GPModel
from ._bessel import periodic_variances


@dataclass
class Periodic(GPModel):
    """Periodic covariance function

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
        """Return harmonic dynamics and full spectral covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Harmonic generator, configured observations/mean, and covariances.
        """
        period, mscale, lscale, sigv, *_ = theta
        variances = periodic_variances(mscale, lscale, self.J, xp=xp)
        return ContinuousMatrices(
            xp.asarray(self._kron) / period, xp.zeros((self.nx, 0)),
            xp.asarray(self.C), xp.asarray(self.D),
            xp.zeros((self.nx, self.nx)), xp.asarray([[sigv**2]]),
            xp.asarray(self.x0), xp.diag(xp.repeat(variances, 2)),
        )

    def _discrete_rotation(self, A: Any, dt: Any, *, xp: Any = np) -> tuple[Any, Any, Any, Any]:
        """Return harmonic rotations and exactly zero input/noise increments.

        Parameters
        ----------
        A : array
            Continuous harmonic generator.
        dt : scalar
            Sampling interval.
        xp : module, optional
            Array namespace.

        Returns
        -------
        tuple
            Transition, two input matrices, and zero diffusion covariance.
        """
        angles = xp.diag(A[1::2, ::2]) * dt
        rotation = xp.kron(xp.diag(xp.cos(angles)), xp.eye(2))
        rotation = rotation + xp.kron(
            xp.diag(xp.sin(angles)), xp.asarray([[0.0, -1.0], [1.0, 0.0]])
        )
        b = xp.zeros((self.nx, self.nu))
        return rotation, b, b, xp.zeros((self.nx, self.nx))

    def discrete_covariance(self, theta: Any, dt: Any, *, xp: Any = np) -> tuple[Any, Any, Any, Any]:
        """Return exact harmonic transitions and zero diffusion covariance.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        dt : scalar
            Sampling interval.
        xp : module, optional
            Array namespace.

        Returns
        -------
        tuple
            Transition, input matrices, and diffusion covariance.
        """
        return self._discrete_rotation(self.continuous_matrices(theta, xp=xp).A, dt, xp=xp)

    def discretization(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return exact rotations and zero diffusion factors for current matrices.

        Parameters
        ----------
        dt : float
            Sampling interval.

        Returns
        -------
        tuple of numpy.ndarray
            Transition, input matrices, and the zero diffusion factor.
        """
        return self._discrete_rotation(self.A, dt)
