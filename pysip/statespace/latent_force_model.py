"""Thermal models augmented by Gaussian-process latent heat inputs."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..params import Parameters
from ._functional import ContinuousMatrices, block_diag
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

    def _combine_continuous(self, rc: ContinuousMatrices, gp: ContinuousMatrices, *, xp: Any = np) -> ContinuousMatrices:
        """Replace the selected RC input by a latent GP observation.

        Parameters
        ----------
        rc, gp : ContinuousMatrices
            Thermal and latent-process covariance realizations.
        xp : module, optional
            Array namespace.

        Returns
        -------
        ContinuousMatrices
            Augmented realization retaining only nonlatent external inputs.
        """
        nr, ng = rc.A.shape[0], gp.A.shape[0]
        a = xp.concatenate((
            xp.concatenate((rc.A, rc.B[:, self.idx] @ gp.C), axis=1),
            xp.concatenate((xp.zeros((ng, nr)), gp.A), axis=1),
        ), axis=0)
        b = xp.concatenate((rc.B[:, ~self.idx], xp.zeros((ng, self.nu))), axis=0)
        c = xp.concatenate((rc.C, rc.D[:, self.idx] @ gp.C), axis=1)
        return ContinuousMatrices(
            a, b, c, rc.D[:, ~self.idx],
            block_diag(rc.Qc, gp.Qc, xp=xp), rc.Rc,
            xp.concatenate((rc.x0, gp.x0), axis=0),
            block_diag(rc.P0c, gp.P0c, xp=xp),
        )

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Evaluate the augmented covariance realization without mutating children.

        Parameters
        ----------
        theta : array
            Physical parameters in thermal-then-GP flattened order.
        xp : module, optional
            Array namespace.

        Returns
        -------
        ContinuousMatrices
            Coupled continuous realization with full covariances.
        """
        split = len(self._rc.params)
        return self._combine_continuous(
            self._rc.continuous_matrices(theta[:split], xp=xp),
            self._gp.continuous_matrices(theta[split:], xp=xp), xp=xp,
        )


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

    def continuous_matrices(self, theta: Any, *, xp: Any = np) -> ContinuousMatrices:
        """Return coupled thermal/Matérn dynamics and full covariances.

        Parameters
        ----------
        theta : array
            Physical parameters in model order.
        xp : module, optional
            NumPy or JAX NumPy array namespace.

        Returns
        -------
        ContinuousMatrices
            Coupled dynamics, configured observations, initial mean/covariances.
        """
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
        ) = theta

        a = xp.asarray([
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 1.0 / Cw, 0.0],
            [1.0 / (Ci * Ri), -1.0 / (Ci * Ri), 1.0 / Ci, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -3.0 / lscale**2, -2.0 * 3.0**0.5 / lscale],
        ])
        b = xp.asarray([
            [1.0 / (Cw * Ro), 0.0], [0.0, 1.0 / Ci],
            [0.0, 0.0], [0.0, 0.0],
        ])
        qc = xp.diag(xp.asarray([
            sigw_w**2, sigw_i**2, 0.0,
            4.0 * 3.0**1.5 * mscale**2 / lscale**3,
        ]))
        p0 = xp.diag(xp.asarray([
            sigx0_w**2, sigx0_i**2, mscale**2, 3.0 * mscale**2 / lscale**2,
        ]))
        x0 = xp.concatenate((xp.asarray([[x0_w], [x0_i]]), xp.asarray(self.x0[2:])), axis=0)
        return ContinuousMatrices(
            a, b, xp.asarray(self.C), xp.asarray(self.D), qc,
            xp.asarray([[sigv**2]]), x0, p0,
        )
