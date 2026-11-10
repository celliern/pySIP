"""Single indoor capacity with outdoor exchange, solar, HVAC, and scaled ventilation heat."""

from dataclasses import dataclass
from typing import cast

from ...params import Parameters
from .._functional import array_set
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

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
        R, C, A, cv, sigw, sigv, x0, sigx0, *_ = cast(Parameters, self.parameters).theta

        self.A = array_set(self.A, (0, 0), -1.0 / (C * R))
        self.B = array_set(self.B, (0, slice(None)), [1.0 / (C * R), A / C, 1.0 / C, cv / C])
        self.Q = array_set(self.Q, (0, 0), sigw)
        self.R = array_set(self.R, (0, 0), sigv)
        self.x0 = array_set(self.x0, (0, 0), x0)
        self.P0 = array_set(self.P0, (0, 0), sigx0)
