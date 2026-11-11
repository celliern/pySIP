"""Indoor/heater capacities linked by Rh, with outdoor exchange through Rw and indoor solar gains."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import RCModel


@dataclass
class TiTh_RwRhAi(RCModel):
    """Indoor/heater capacities linked by Rh, with outdoor exchange through Rw and indoor solar gains."""

    states = (
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xh", "heater temperature"),
    )

    params = (
        ("THERMAL_RESISTANCE", "Rw", "between the outdoor and the indoor"),
        ("THERMAL_RESISTANCE", "Rh", "between the indoor and the heater"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Ch", "of the heater"),
        ("SOLAR_APERTURE", "Ai", "effective solar aperture"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_h", "of the heater dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_h", "of the heater temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_h", "of the heater temperature"),
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
            Rh,
            Ci,
            Ch,
            Ai,
            sigw_i,
            sigw_h,
            sigv,
            x0_i,
            x0_h,
            sigx0_i,
            sigx0_h,
            *_,
        ) = theta

        A_matrix = xp.array([
            [-(Rh + Rw) / (Ci * Rh * Rw), 1.0 / (Ci * Rh)],
            [1.0 / (Ch * Rh), -1.0 / (Ch * Rh)],
        ])
        B_matrix = xp.array([
            [1.0 / (Ci * Rw), Ai / Ci, 0.0],
            [0.0, 0.0, 1.0 / Ch],
        ])
        return ContinuousMatrices(
            A=A_matrix,
            B=B_matrix,
            C=xp.asarray(self.C),
            D=xp.asarray(self.D),
            Qc=xp.diag(xp.array([sigw_i ** 2, sigw_h ** 2])),
            Rc=xp.array([[sigv ** 2]]),
            x0=xp.array([x0_i, x0_h])[:, None],
            P0c=xp.diag(xp.array([sigx0_i ** 2, sigx0_h ** 2])),
        )
