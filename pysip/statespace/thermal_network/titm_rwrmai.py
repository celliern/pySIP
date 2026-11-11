"""Indoor/internal-mass capacities linked by Rm, with outdoor exchange through Rw and indoor solar gains."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import RCModel


@dataclass
class TiTm_RwRmAi(RCModel):
    """Indoor/internal-mass capacities linked by Rm, with outdoor exchange through Rw and indoor solar gains."""

    states = (
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xm", "internal mass temperature"),
    )

    params = (
        ("THERMAL_RESISTANCE", "Rw", "between the outdoor and the indoor"),
        ("THERMAL_RESISTANCE", "Rm", "between the indoor and the internal mass"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Cm", "of the internal mass"),
        ("SOLAR_APERTURE", "Ai", "effective solar aperture"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_m", "of the internal mass dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_m", "of the internal mass temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_m", "of the internal mass temperature"),
    )

    inputs = (
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
    )

    outputs = (("TEMPERATURE", "xi", "indoor temperature"),)

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self.C[0, 0] = 1.0

    def continuous_matrices(
        self, theta: Any, *, xp: Any = np
    ) -> ContinuousMatrices:
        """Return continuous matrices and full covariances without mutation.

        Parameters
        ----------
        theta : array_like
            Physical parameters in model order.
        xp : module, optional
            NumPy-compatible array namespace.

        Returns
        -------
        ContinuousMatrices
            System matrices, full covariances, and initial mean.
        """
        (
            Rw,
            Rm,
            Ci,
            Cm,
            Ai,
            sigw_i,
            sigw_m,
            sigv,
            x0_i,
            x0_m,
            sigx0_i,
            sigx0_m,
            *_,
        ) = theta

        A_matrix = xp.array([
            [-(Rm + Rw) / (Ci * Rm * Rw), 1.0 / (Ci * Rm)],
            [1.0 / (Cm * Rm), -1.0 / (Cm * Rm)],
        ])
        B_matrix = xp.array([
            [1.0 / (Ci * Rw), Ai / Ci, 1.0 / Ci],
            [0.0, 0.0, 0.0],
        ])
        return ContinuousMatrices(
            A=A_matrix,
            B=B_matrix,
            C=xp.asarray(self.C),
            D=xp.asarray(self.D),
            Qc=xp.diag(xp.array([sigw_i ** 2, sigw_m ** 2])),
            Rc=xp.array([[sigv ** 2]]),
            x0=xp.array([x0_i, x0_m])[:, None],
            P0c=xp.diag(xp.array([sigx0_i ** 2, sigx0_m ** 2])),
        )
