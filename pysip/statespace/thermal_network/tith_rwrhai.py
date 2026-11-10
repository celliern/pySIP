"""Indoor/heater capacities linked by Rh, with outdoor exchange through Rw and indoor solar gains."""

from dataclasses import dataclass
from typing import cast

from ...params import Parameters
from .._functional import array_set
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

    def update_continuous_ssm(self) -> None:
        """Mutate continuous matrices and initial moments from physical parameters."""
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
        ) = cast(Parameters, self.parameters).theta

        self.A = array_set(self.A, slice(None), [
            [-(Rh + Rw) / (Ci * Rh * Rw), 1.0 / (Ci * Rh)],
            [1.0 / (Ch * Rh), -1.0 / (Ch * Rh)],
        ])
        self.B = array_set(self.B, slice(None), [[1.0 / (Ci * Rw), Ai / Ci, 0.0], [0.0, 0.0, 1.0 / Ch]])
        self.Q = array_set(self.Q, self._diag, [sigw_i, sigw_h])
        self.R = array_set(self.R, (0, 0), sigv)
        self.x0 = array_set(self.x0, (slice(None), 0), [x0_i, x0_h])
        self.P0 = array_set(self.P0, self._diag, [sigx0_i, sigx0_h])
