"""Block state-space realization of independent temporal kernel sums."""

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg import block_diag

from ...params import Parameters
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

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        self._gp1.update_continuous_ssm()
        self._gp2.update_continuous_ssm()

        self.A[: self._gp1.nx, : self._gp1.nx] = self._gp1.A
        self.A[self._gp1.nx :, self._gp1.nx :] = self._gp2.A

        self.C[:, : self._gp1.nx] = self._gp1.C
        self.C[:, self._gp1.nx :] = self._gp2.C

        self.Q[: self._gp1.nx, : self._gp1.nx] = self._gp1.Q
        self.Q[self._gp1.nx :, self._gp1.nx :] = self._gp2.Q

        self.R = self._gp1.R

        self.P0[: self._gp1.nx, : self._gp1.nx] = self._gp1.P0
        self.P0[self._gp1.nx :, self._gp1.nx :] = self._gp2.P0

    def discretization(
        self, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compose independent discrete covariances without stationary subtraction."""
        a1, _, _, q1 = self._gp1.discretization(dt)
        a2, _, _, q2 = self._gp2.discretization(dt)
        b = self.B.copy()
        return (
            cast(np.ndarray, block_diag(a1, a2)), b, b,
            cast(np.ndarray, block_diag(q1, q2)),
        )
