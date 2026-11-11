"""Envelope/indoor capacities in an outdoor-Ro-envelope-Ri-indoor chain, with indoor HVAC heat."""

from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import RCModel


class TwTi_RoRi(RCModel):
    """Envelope/indoor capacities in an outdoor-Ro-envelope-Ri-indoor chain, with indoor HVAC heat."""

    states = (
        ("TEMPERATURE", "xw", "envelope temperature"),
        ("TEMPERATURE", "xi", "indoor temperature"),
    )

    params = (
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the envelope"),
        ("THERMAL_RESISTANCE", "Ri", "between the envelope and the indoor"),
        ("THERMAL_CAPACITY", "Cw", "of the envelope"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("STATE_DEVIATION", "sigw_w", "of the envelope dynamic"),
        ("STATE_DEVIATION", "sigw_i", "of the indoor dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0_w", "of the envelope temperature"),
        ("INITIAL_MEAN", "x0_i", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0_w", "of the envelope temperature"),
        ("INITIAL_DEVIATION", "sigx0_i", "of the infoor temperature"),
    )

    inputs = (
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qh", "HVAC system heat"),
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
            [1.0 / (Cw * Ro), 0.0],
            [0.0, 1.0 / Ci],
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
