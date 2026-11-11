"""Block state-space realization of independent temporal kernel sums."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ...params import Parameters
from ...utils.math import nearest_cholesky
from .._functional import ContinuousMatrices, block_diag
from ..base import GPModel, Node, Par


@dataclass
class GPSum(GPModel):
    """Sum of two Gaussian Process model

    Parameters
    ----------
    gp1 : GPModel
        GPModel instance
    gp2 : GPModel
        GPModel instance

    Notes
    -----
    The MEASURE_DEVIATION of the `gp2` is fixed
    because it is already defined in `gp1`.
    """

    def __init__(self, gp1: GPModel, gp2: GPModel) -> None:
        """Compose deep copies without mutating the supplied models."""
        if not isinstance(gp1, GPModel):
            raise TypeError("`gp1` must be an GPModel instance")

        if not isinstance(gp2, GPModel):
            raise TypeError("`gp2` must be an GPModel instance")

        self._gp1 = deepcopy(gp1)
        self._gp2 = deepcopy(gp2)
        # Both copies have completed StateSpace.__post_init__.
        parameters1 = cast(Parameters, self._gp1.parameters)
        parameters2 = cast(Parameters, self._gp2.parameters)

        for node in cast(list[Node], self._gp2.params):
            if node.category == Par.MEASURE_DEVIATION:
                parameters2.set_parameter(
                    node.name, value=0.0, transform="fixed"
                )
                break

        self.parameters = parameters1 + parameters2

        _states = cast(list[Node], self._gp1.states) + cast(list[Node], self._gp2.states)
        self.states = [s.unpack() for s in _states]

        self.params = [
            (node.category.name, name, node.description)
            for node, name in zip(
                [*cast(list[Node], self._gp1.params), *cast(list[Node], self._gp2.params)],
                self.parameters.ids,
            )
        ]

        self.inputs = []

        self.outputs = [("ANY", "sum(f(t))", "sum of stochastic processes")]

        self.name = self._gp1.name + "__+__" + self._gp2.name

        super().__post_init__()

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self._gp1.set_constant_continuous_ssm()
        self._gp2.set_constant_continuous_ssm()

    def _combine_continuous(self, left: ContinuousMatrices, right: ContinuousMatrices, *, xp: Any = np) -> ContinuousMatrices:
        """Combine independent child covariance realizations.

        Parameters
        ----------
        left, right : ContinuousMatrices
            Child continuous covariance realizations.
        xp : module, optional
            Array namespace.

        Returns
        -------
        ContinuousMatrices
            Block-diagonal realization with summed observations.
        """
        nx = left.A.shape[0] + right.A.shape[0]
        return ContinuousMatrices(
            block_diag(left.A, right.A, xp=xp), xp.zeros((nx, 0)),
            xp.concatenate((left.C, right.C), axis=1),
            xp.zeros((left.C.shape[0], 0)),
            block_diag(left.Qc, right.Qc, xp=xp), left.Rc,
            xp.asarray(self.x0),
            block_diag(left.P0c, right.P0c, xp=xp),
        )

    @staticmethod
    def _combine_discrete(a1: Any, q1: Any, a2: Any, q2: Any, *, xp: Any = np) -> tuple[Any, Any, Any, Any]:
        """Combine independent transitions and diffusion covariance increments.

        Parameters
        ----------
        a1, a2 : array
            Child transitions.
        q1, q2 : array
            Child discrete diffusion covariances.
        xp : module, optional
            Array namespace.

        Returns
        -------
        tuple
            Transition, empty input matrices, and diffusion covariance.
        """
        a = block_diag(a1, a2, xp=xp)
        b = xp.zeros((a.shape[0], 0))
        return a, b, b, block_diag(q1, q2, xp=xp)

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Evaluate a sum covariance realization without mutating either child.

        Parameters
        ----------
        theta : array
            Physical parameters in flattened child order.
        xp : module, optional
            Array namespace.

        Returns
        -------
        ContinuousMatrices
            Independent sum realization with full covariances.
        """
        split = len(self._gp1.params)
        return self._combine_continuous(
            self._gp1.continuous_matrices(theta[:split], xp=xp),
            self._gp2.continuous_matrices(theta[split:], xp=xp), xp=xp,
        )

    def discrete_covariance(self, theta: Any, dt: Any, *, xp: Any = np) -> tuple[Any, Any, Any, Any]:
        """Recursively evaluate independent diffusion covariance increments.

        Parameters
        ----------
        theta : array
            Physical parameters in flattened child order.
        dt : scalar
            Sampling interval.
        xp : module, optional
            Array namespace.

        Returns
        -------
        tuple
            Transition, input matrices, and diffusion covariance.
        """
        split = len(self._gp1.params)
        a1, _, _, q1 = self._gp1.discrete_covariance(theta[:split], dt, xp=xp)
        a2, _, _, q2 = self._gp2.discrete_covariance(theta[split:], dt, xp=xp)
        return self._combine_discrete(a1, q1, a2, q2, xp=xp)

    def discretization(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Factor the shared recursive independent covariance law.

        Parameters
        ----------
        dt : float
            Sampling interval.

        Returns
        -------
        tuple of numpy.ndarray
            Transition, input matrices, and upper diffusion factor.
        """
        a1, _, _, q1 = self._gp1.discretization(dt)
        a2, _, _, q2 = self._gp2.discretization(dt)
        a, b0, b1, q = self._combine_discrete(a1, q1.T @ q1, a2, q2.T @ q2)
        return a, b0, b1, cast(np.ndarray, nearest_cholesky(q))
