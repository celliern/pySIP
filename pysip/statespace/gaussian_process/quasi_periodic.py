"""Periodic kernels modulated by Matérn temporal decay."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from ...params import Parameters
from .._functional import array_namespace, array_set
from ..base import GPModel
from ._bessel import periodic_scales


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
        self.C = array_set(self.C, (0, slice(None, None, 2)), 1.0)
        self._kron = np.kron(
            np.diag(range(self.J + 1)),
            np.array([[0.0, -2.0 * np.pi], [2.0 * np.pi, 0.0]]),
        )

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        period, mscale, lscale, sigv, decay, *_ = cast(Parameters, self.parameters).theta

        xp = array_namespace(self.A)
        self.A = array_set(self.A, slice(None), self._kron / period + xp.kron(
            xp.eye(self.J + 1), xp.asarray([[-1.0 / decay, 0.0], [0.0, -1.0 / decay]])
        ))
        scales = periodic_scales(mscale, lscale, self.J, xp=xp)
        self.R = array_set(self.R, (0, 0), sigv)
        self.P0 = array_set(self.P0, self._diag, xp.repeat(scales, 2))
        self.Q = array_set(self.Q, slice(None), self.P0 * 2.0**0.5 * decay ** (-0.5))


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
        self.C = array_set(self.C, (0, slice(None, None, 4)), 1.0)
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

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        period, mscale, lscale, sigv, decay, *_ = cast(Parameters, self.parameters).theta

        xp = array_namespace(self.A)
        tmp1 = -3.0 / decay**2
        tmp2 = -2.0 * 3.0**0.5 / decay
        self.A = array_set(self.A, slice(None), self._kron / period + xp.kron(
            xp.eye(self.J + 1),
            xp.asarray(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [tmp1, tmp2, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, tmp1, tmp2],
                ]
            ),
        ))
        scales = periodic_scales(mscale, lscale, self.J, xp=xp)
        self.R = array_set(self.R, (0, 0), sigv)
        _P0 = xp.kron(xp.diag(scales), xp.eye(2))
        self.Q = array_set(self.Q, slice(None), xp.kron(
            _P0, xp.asarray([[0.0, 0.0], [0.0, 2.0 * 3.0**0.75 / decay**1.5]])
        ))
        self.P0 = array_set(self.P0, slice(None), xp.kron(
            _P0, xp.asarray([[1.0, 0.0], [0.0, 3.0**0.5 / decay]])
        ))
