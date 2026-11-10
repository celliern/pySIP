"""Kronecker state-space realization of a product of temporal kernels."""

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, cast

import numpy as np

from ...params import Parameters
from ...utils.math import nearest_cholesky
from .._functional import ContinuousMatrices, apply_covariances, model_covariances
from ..base import GPModel, Node, Par


@dataclass
class GPProduct(GPModel):
    """Product of two Gaussian Process Covariance

    Parameters
    ----------
    gp1 : GPModel
        GPModel instance
    gp2 : GPModel
        GPModel instance

    Notes
    -----
    The MEASURE_DEVIATION and MAGNITUDE_SCALE of the `gp2` are fixed
    because they are already defined in `gp1`.

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
            if node.category == Par.MAGNITUDE_SCALE:
                parameters2.set_parameter(
                    node.name, value=1.0, transform="fixed"
                )
            if node.category == Par.MEASURE_DEVIATION:
                parameters2.set_parameter(
                    node.name, value=0.0, transform="fixed"
                )

        self.parameters = parameters1 + parameters2

        states: list[tuple[str, str, str]] = []
        for s1, s2 in product(
            cast(list[Node], self._gp1.states),
            cast(list[Node], self._gp2.states),
        ):
            state = (
                s1.category.name,
                s1.name + "*" + s2.name,
                s1.description + "*" + s2.description,
            )
            states.append(state)
        self.states = states

        self.name = self._gp1.name + "__x__" + self._gp2.name

        self.params = [
            (node.category.name, name, node.description)
            for node, name in zip(
                [*cast(list[Node], self._gp1.params), *cast(list[Node], self._gp2.params)],
                self.parameters.ids,
            )
        ]

        self.inputs = []

        if np.sum(self._gp1.C.squeeze()) + np.sum(self._gp2.C.squeeze()) > 2:
            self.outputs = [("ANY", "sum(f(t))", "sum of stochastic processes")]
        else:
            self.outputs = [("ANY", "f(t)", "stochastic processes")]

        super().__post_init__()

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self._gp1.set_constant_continuous_ssm()
        self._gp2.set_constant_continuous_ssm()

    def _combine_continuous(self, left: ContinuousMatrices, right: ContinuousMatrices, *, xp: Any = np) -> ContinuousMatrices:
        """Combine child covariance realizations using the product kernel law.

        Parameters
        ----------
        left, right : ContinuousMatrices
            Child continuous covariance realizations.
        xp : module, optional
            Array namespace.

        Returns
        -------
        ContinuousMatrices
            Kronecker product realization with full covariances.
        """
        nx1, nx2 = left.A.shape[0], right.A.shape[0]
        return ContinuousMatrices(
            xp.kron(left.A, xp.eye(nx2)) + xp.kron(xp.eye(nx1), right.A),
            xp.zeros((nx1 * nx2, 0)),
            xp.kron(left.C, right.C),
            xp.zeros((left.C.shape[0] * right.C.shape[0], 0)),
            xp.kron(left.Qc, right.P0c) + xp.kron(left.P0c, right.Qc),
            left.Rc,
            xp.asarray(self.x0),
            xp.kron(left.P0c, right.P0c),
        )

    @staticmethod
    def _combine_discrete(a1: Any, q1: Any, p1: Any, a2: Any, q2: Any, p2: Any, *, xp: Any = np) -> tuple[Any, Any, Any, Any]:
        """Combine positive covariance increments without stationary subtraction.

        Parameters
        ----------
        a1, a2 : array
            Child transitions.
        q1, q2 : array
            Child discrete diffusion covariances.
        p1, p2 : array
            Child stationary covariances.
        xp : module, optional
            Array namespace.

        Returns
        -------
        tuple
            Transition, empty input matrices, and product diffusion covariance.
        """
        a = xp.kron(a1, a2)
        q = xp.kron(q1, p2) + xp.kron(a1 @ p1 @ a1.T, q2)
        b = xp.zeros((a.shape[0], 0))
        return a, b, b, q

    def update_continuous_ssm(self) -> None:
        """Refresh child realizations and factor the shared product covariance law."""
        self._gp1.update_continuous_ssm()
        self._gp2.update_continuous_ssm()
        apply_covariances(self, self._combine_continuous(
            model_covariances(self._gp1), model_covariances(self._gp2)
        ))

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Evaluate the product covariance realization without mutating either child.

        Parameters
        ----------
        theta : array
            Physical parameters in flattened child order.
        xp : module, optional
            Array namespace.

        Returns
        -------
        ContinuousMatrices
            Product realization with full covariances, never QR factors.
        """
        split = len(self._gp1.params)
        return self._combine_continuous(
            self._gp1.continuous_matrices(theta[:split], xp=xp),
            self._gp2.continuous_matrices(theta[split:], xp=xp), xp=xp,
        )

    def discrete_covariance(self, theta: Any, dt: Any, *, xp: Any = np) -> tuple[Any, Any, Any, Any]:
        """Recursively evaluate product transition and diffusion covariance.

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
        theta1, theta2 = theta[:split], theta[split:]
        a1, _, _, q1 = self._gp1.discrete_covariance(theta1, dt, xp=xp)
        a2, _, _, q2 = self._gp2.discrete_covariance(theta2, dt, xp=xp)
        return self._combine_discrete(
            a1, q1, self._gp1.continuous_matrices(theta1, xp=xp).P0c,
            a2, q2, self._gp2.continuous_matrices(theta2, xp=xp).P0c, xp=xp,
        )

    def discretization(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Factor the same recursive covariance law used by the pure evaluator.

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
        a, b0, b1, q = self._combine_discrete(
            a1, q1.T @ q1, self._gp1.P0.T @ self._gp1.P0,
            a2, q2.T @ q2, self._gp2.P0.T @ self._gp2.P0,
        )
        return a, b0, b1, cast(np.ndarray, nearest_cholesky(q))
