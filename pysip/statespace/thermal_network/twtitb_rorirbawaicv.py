"""Envelope, indoor, and boundary-wall capacities with solar and scaled ventilation heat."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import RCModel


@dataclass
class TwTiTb_RoRiRbAwAicv(RCModel):
    """Envelope, indoor, and boundary-wall capacities with solar and scaled ventilation heat."""

    states = (
        ("TEMPERATURE", "xw", "envelope temperature"),
        ("TEMPERATURE", "xi", "indoor temperature"),
        ("TEMPERATURE", "xb", "boundary wall temperature"),
    )

    params = (
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the envelope"),
        ("THERMAL_RESISTANCE", "Ri", "between the envelope and the indoor"),
        ("THERMAL_RESISTANCE", "Rb", "between the indoor and the boundary"),
        ("THERMAL_CAPACITY", "Cw", "of the envelope"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("THERMAL_CAPACITY", "Cb", "of the wall between the indoor and the boundary"),
        ("SOLAR_APERTURE", "Aw", "of the envelope"),
        ("SOLAR_APERTURE", "Ai", "of the windows"),
        ("COEFFICIENT", "cv", "scaling of the heat from the ventilation"),
        ("STATE_DEVIATION", "sigw_w", "of the envelope dynamic"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("STATE_DEVIATION", "sigw_b", "of the boundary wall dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_w", "of the envelope temperature"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_MEAN", "x0_b", "of the boundary wall temperature"),
        ("INITIAL_DEVIATION", "sigx0_w", "of the envelope temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_b", "of the boundary wall temperature"),
    )

    inputs = (
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("TEMPERATURE", "Tb", "boundary temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
        ("POWER", "Qv", "heat from the ventilation system"),
    )

    outputs = (("TEMPERATURE", "xi", "indoor temperature"),)

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
            Rb,
            Cw,
            Ci,
            Cb,
            Aw,
            Ai,
            cv,
            sigw_w,
            sigw_i,
            sigw_b,
            sigv,
            x0_w,
            x0_i,
            x0_b,
            sigx0_w,
            sigx0_i,
            sigx0_b,
            *_,
        ) = theta

        A_matrix = xp.array([
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri), 0.0],
            [1.0 / (Ci * Ri), -(Rb + 2.0 * Ri) / (Ci * Rb * Ri), 2.0 / (Ci * Rb)],
            [0.0, 2.0 / (Cb * Rb), -4.0 / (Cb * Rb)],
        ])
        B_matrix = xp.array([
            [1.0 / (Cw * Ro), 0.0, Aw / Cw, 0.0, 0.0],
            [0.0, 0.0, Ai / Ci, 1.0 / Ci, cv / Ci],
            [0.0, 2.0 / (Cb * Rb), 0.0, 0.0, 0.0],
        ])
        return ContinuousMatrices(
            A=A_matrix,
            B=B_matrix,
            C=xp.asarray(self.C),
            D=xp.asarray(self.D),
            Qc=xp.diag(xp.array([sigw_w ** 2, sigw_i ** 2, sigw_b ** 2])),
            Rc=xp.array([[sigv ** 2]]),
            x0=xp.array([x0_w, x0_i, x0_b])[:, None],
            P0c=xp.diag(xp.array([sigx0_w ** 2, sigx0_i ** 2, sigx0_b ** 2])),
        )
