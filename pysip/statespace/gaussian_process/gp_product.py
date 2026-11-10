"""Kronecker state-space realization of a product of temporal kernels."""

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import cast

import numpy as np

from ...params import Parameters
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

        self._I1 = np.eye(self._gp1.nx)
        self._I2 = np.eye(self._gp2.nx)

        super().__post_init__()

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self._gp1.set_constant_continuous_ssm()
        self._gp2.set_constant_continuous_ssm()

    def update_continuous_ssm(self) -> None:
        """Refresh product dynamics and factor the sum of child diffusion rates."""
        self._gp1.update_continuous_ssm()
        self._gp2.update_continuous_ssm()

        self.A = np.kron(self._gp1.A, self._I2) + np.kron(self._I1, self._gp2.A)
        self.C = np.kron(self._gp1.C, self._gp2.C)
        self.Q = np.linalg.qr(
            np.vstack((
                np.kron(self._gp1.Q, self._gp2.P0),
                np.kron(self._gp1.P0, self._gp2.Q),
            )),
            mode="r",
        )
        self.R = self._gp1.R
        self.P0 = np.kron(self._gp1.P0, self._gp2.P0)

    def discretization(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compose product covariance from positive semidefinite child increments.

        P1⊗P2 - (A1 P1 A1.T)⊗(A2 P2 A2.T) equals
        Q1⊗P2 + (A1 P1 A1.T)⊗Q2, without stationary subtraction.
        """
        a1, _, _, q1 = self._gp1.discretization(dt)
        a2, _, _, q2 = self._gp2.discretization(dt)
        q = np.linalg.qr(
            np.vstack((
                np.kron(q1, self._gp2.P0),
                np.kron(self._gp1.P0 @ a1.T, q2),
            )),
            mode="r",
        )
        b = np.zeros((self.nx, self.nu))
        return np.kron(a1, a2), b, b, q
