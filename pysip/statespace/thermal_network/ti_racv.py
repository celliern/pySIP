"""Single indoor capacity with outdoor exchange, solar, HVAC, and scaled ventilation heat."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .._functional import ContinuousMatrices
from ..base import RCModel


@dataclass
class Ti_RAcv(RCModel):
    """Single indoor capacity with outdoor exchange, solar, HVAC, and scaled ventilation heat."""
    states = (("TEMPERATURE", "xi", "indoor temperature"),)

    params = (
        ("THERMAL_RESISTANCE", "R", "between the outdoor and the indoor"),
        ("THERMAL_CAPACITY", "C", "effective overall capacity"),
        ("SOLAR_APERTURE", "A", "effective solar aperture"),
        ("COEFFICIENT", "cv", "scaling of the heat from the ventilation"),
        ("STATE_DEVIATION", "sigw", "of the indoor dynamic"),
        ("MEASURE_DEVIATION", "sigv", "of the indoor temperature measurements"),
        ("INITIAL_MEAN", "x0", "of the infoor temperature"),
        ("INITIAL_DEVIATION", "sigx0", "of the infoor temperature"),
    )

    inputs = (
        ("TEMPERATURE", "To", "outdoor temperature"),
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
        ("POWER", "Qv", "heat from the ventilation system"),
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
        R, C, A, cv, sigw, sigv, x0, sigx0, *_ = theta

        A_matrix = xp.array([
            [-1.0 / (C * R)],
        ])
        B_matrix = xp.array([
            [1.0 / (C * R), A / C, 1.0 / C, cv / C],
        ])
        return ContinuousMatrices(
            A=A_matrix,
            B=B_matrix,
            C=xp.asarray(self.C),
            D=xp.asarray(self.D),
            Qc=xp.diag(xp.array([sigw ** 2])),
            Rc=xp.array([[sigv ** 2]]),
            x0=xp.array([x0])[:, None],
            P0c=xp.diag(xp.array([sigx0 ** 2])),
        )
