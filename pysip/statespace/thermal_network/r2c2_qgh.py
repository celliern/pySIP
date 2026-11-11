"""Two-capacity wall/indoor model with solar heat entering both nodes."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import RCModel


@dataclass
class R2C2Qgh(RCModel):
    """Two-capacity wall/indoor model with solar heat entering both nodes."""
    states = (
        ("TEMPERATURE", "xw", "wall temperature"),
        ("TEMPERATURE", "xi", "indoor space temperature"),
    )

    params = (
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the wall node"),
        ("THERMAL_RESISTANCE", "Ri", "between the wall node and the indoor"),
        ("THERMAL_CAPACITY", "Cw", "Wall"),
        ("THERMAL_CAPACITY", "Ci", "indoor air, indoor walls, furnitures, etc. "),
        ("STATE_DEVIATION", "sigw_w", ""),
        ("STATE_DEVIATION", "sigw_i", ""),
        ("MEASURE_DEVIATION", "sigv", ""),
        ("INITIAL_MEAN", "x0_w", ""),
        ("INITIAL_MEAN", "x0_i", ""),
        ("INITIAL_DEVIATION", "sigx0_w", ""),
        ("INITIAL_DEVIATION", "sigx0_i", ""),
    )

    inputs = (
        ("TEMPERATURE", "To", "outdoor air temperature"),
        ("POWER", "Qgh", "global horizontal solar radiation"),
        ("POWER", "Qh", "HVAC system heat"),
    )

    outputs = (("TEMPERATURE", "xi", "indoor air temperature"),)

    def set_constant_continuous_ssm(self) -> None:
        """Mutate parameter-independent continuous matrices in place."""
        self.C[0, 1] = 1.0

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
            *_,
        ) = theta

        A_matrix = xp.array([
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri)],
            [1.0 / (Ci * Ri), -1.0 / (Ci * Ri)],
        ])
        B_matrix = xp.array([
            [1.0 / (Cw * Ro), 1.0 / Cw, 0.0],
            [0.0, 1.0 / Ci, 1.0 / Ci],
        ])
        return ContinuousMatrices(
            A=A_matrix,
            B=B_matrix,
            C=xp.asarray(self.C),
            D=xp.asarray(self.D),
            Qc=xp.diag(xp.array([sigw_w ** 2, sigw_i ** 2])),
            Rc=xp.array([[sigv ** 2]]),
            x0=xp.array([x0_w, x0_i])[:, None],
            P0c=xp.diag(xp.array([sigx0_w ** 2, sigx0_i ** 2])),
        )
