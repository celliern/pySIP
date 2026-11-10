"""Thermal models augmented by Gaussian-process latent heat inputs."""

from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import numpy as np

from ..params import Parameters
from .base import GPModel, Node, Par, RCModel, StateSpace


@dataclass
class LatentForceModel(StateSpace):
    """Latent Force Model (LFM)

    Parameters
    ----------
    rc : RCModel
        RCModel instance
    gp : GPModel
        GPModel instance
    latent_force : str or list[str]
        Exactly one input name to treat as a latent force.

    Notes
    -----
    The MEASURE_DEVIATION of the GPModel is fixed because it is not used in the
    latent force model. The GPModel is augmented into the RCModel, therefore, only
    the measurement noise matrix `R` of the RCModel is used. To avoid useless
    computation, the MEASURE_DEVIATION of the GPModel must stay fixed.
    """

    def __init__(self, rc: RCModel, gp: GPModel, latent_force: str | list[str]) -> None:
        """Copy both models and replace exactly one matching RC input.

        Raise TypeError for invalid model/input types and ValueError for absent
        RC inputs or a latent-force list that does not name exactly one input.
        The copied GP measurement deviation is fixed to zero.
        """
        if not isinstance(rc, RCModel):
            raise TypeError("`rc` must be an RCModel instance")

        if not isinstance(gp, GPModel):
            raise TypeError("`gp` must be an GPModel instance")

        if not isinstance(latent_force, (str, list)):
            raise TypeError("`latent_force` must be a string or a list of strings")

        if isinstance(latent_force, str):
            latent_force = [latent_force]
        if len(latent_force) != 1:
            raise ValueError("Exactly one latent-force input is supported")

        if not rc.inputs:
            raise ValueError("The `rc` model has no input")

        index = []
        # Model operands have already normalized their instance metadata.
        for i, node in enumerate(cast(list[Node], rc.inputs)):
            if node.name in latent_force:
                index.append(i)
        if not index:
            raise ValueError("`latent_force` is not an input of `rc`")

        self._rc = deepcopy(rc)
        self._gp = deepcopy(gp)
        parameters_rc = cast(Parameters, self._rc.parameters)
        parameters_gp = cast(Parameters, self._gp.parameters)

        # The GP have only one output
        for i, node in enumerate(cast(list[Node], self._gp.params)):
            if node.category == Par.MEASURE_DEVIATION:
                parameters_gp.set_parameter(
                    node.name, value=0.0, transform="fixed"
                )
                break

        self.parameters = parameters_rc + parameters_gp
        self.states = cast(list[Node], self._rc.states) + cast(list[Node], self._gp.states)

        self.params = [
            (node.category.name, name, node.description)
            for node, name in zip(
                [*cast(list[Node], self._rc.params), *cast(list[Node], self._gp.params)],
                self.parameters.ids,
            )
        ]

        self.inputs = [*cast(list[Node], self._rc.inputs)]
        for i in sorted(index, reverse=True):
            del self.inputs[i]
        self.outputs = [*cast(list[Node], self._rc.outputs)]

        # Unpack list of Node as a list of string to rebuild list of Node
        self.states = [s.unpack() for s in cast(list[Node], self.states)]
        self.inputs = [s.unpack() for s in cast(list[Node], self.inputs)]
        self.outputs = [s.unpack() for s in cast(list[Node], self.outputs)]

        # slicing columns in the input matrix corresponding to latent forces
        self.idx = np.full((self._rc.nu), False)

        self.idx[index] = True

        self.hold_order = self._rc.hold_order

        self.name = (
            self._rc.name + "__" + "latent_forces(" + self._rc.latent_forces + ")"
        )

        super().__post_init__()

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self._rc.set_constant_continuous_ssm()
        self._gp.set_constant_continuous_ssm()

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        self._rc.update_continuous_ssm()
        self._gp.update_continuous_ssm()

        self.A[: self._rc.nx, : self._rc.nx] = self._rc.A
        self.A[self._rc.nx :, self._rc.nx :] = self._gp.A
        self.A[: self._rc.nx, self._rc.nx :] = self._rc.B[:, self.idx] @ self._gp.C

        self.B[: self._rc.nx, :] = self._rc.B[:, ~self.idx]

        self.C[:, : self._rc.nx] = self._rc.C

        self.D = self._rc.D[:, ~self.idx]

        self.Q[: self._rc.nx, : self._rc.nx] = self._rc.Q
        self.Q[self._rc.nx :, self._rc.nx :] = self._gp.Q

        self.R = self._rc.R

        self.x0[: self._rc.nx] = self._rc.x0
        self.x0[self._rc.nx :] = self._gp.x0

        self.P0[: self._rc.nx, : self._rc.nx] = self._rc.P0
        self.P0[self._rc.nx :, self._rc.nx :] = self._gp.P0


@dataclass
class R2C2_Qgh_Matern32(RCModel):
    """LFM: R2C2_Qgh + Matern32

    The Matérn kernel with smoothness parameter = 3/2 is used to model the
    the solar radiation as a latent force in the model R2C2.

    This LFM is built by hand and allows to check the creation of LFM in
    bopt.Latent_force_model.
    """

    states = (
        ("TEMPERATURE", "xw", "wall temperature"),
        ("TEMPERATURE", "xi", "indoor space temperature"),
        ("ANY", "f(t)", "stochastic process"),
        ("ANY", "df(t)/dt", "derivative stochastic process"),
    )

    params = (
        ("THERMAL_RESISTANCE", "R2C2Qgh__Ro", "between the outdoor and the wall node"),
        ("THERMAL_RESISTANCE", "R2C2Qgh__Ri", "between the wall node and the indoor"),
        ("THERMAL_CAPACITY", "R2C2Qgh__Cw", "Wall"),
        (
            "THERMAL_CAPACITY",
            "R2C2Qgh__Ci",
            "indoor air, indoor walls, furnitures, etc. ",
        ),
        ("STATE_DEVIATION", "R2C2Qgh__sigw_w", ""),
        ("STATE_DEVIATION", "R2C2Qgh__sigw_i", ""),
        ("MEASURE_DEVIATION", "R2C2Qgh__sigv", ""),
        ("INITIAL_MEAN", "R2C2Qgh__x0_w", ""),
        ("INITIAL_MEAN", "R2C2Qgh__x0_i", ""),
        ("INITIAL_DEVIATION", "R2C2Qgh__sigx0_w", ""),
        ("INITIAL_DEVIATION", "R2C2Qgh__sigx0_i", ""),
        ("MAGNITUDE_SCALE", "Matern32__mscale", ""),
        ("LENGTH_SCALE", "Matern32__lscale", ""),
    )

    inputs = (
        ("TEMPERATURE", "To", "outdoor air temperature"),
        ("POWER", "Qh", "HVAC system heat"),
    )

    outputs = (("TEMPERATURE", "xi", "indoor air temperature"),)

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self.C[0, 1] = 1.0

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        (
            Ro,
            Ri,
            Cw,
            Ci,
            sigw_w,
            sigw_i,
            sigv,
            x0_w,
            x0_i,
            sigx0_w,
            sigx0_i,
            mscale,
            lscale,
            *_,
        ) = cast(Parameters, self.parameters).theta

        self.A[:] = [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 1.0 / Cw, 0.0],
            [1.0 / (Ci * Ri), -1.0 / (Ci * Ri), 1.0 / Ci, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -3.0 / lscale**2, -2.0 * 3.0**0.5 / lscale],
        ]
        self.B[:2, :] = [[1.0 / (Cw * Ro), 0.0], [0.0, 1.0 / Ci]]
        self.Q[self._diag] = [
            sigw_w,
            sigw_i,
            0.0,
            2.0 * 3.0**0.75 * mscale / lscale**1.5,
        ]
        self.R[0, 0] = sigv
        self.x0[:2, 0] = [x0_w, x0_i]
        self.P0[self._diag] = [sigx0_w, sigx0_i, mscale, 3.0**0.5 * mscale / lscale]
