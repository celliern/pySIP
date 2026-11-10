"""Envelope/indoor RC chain with separate envelope and indoor solar apertures."""

from dataclasses import dataclass
from typing import cast

from ...params import Parameters
from .._functional import array_set
from ..base import RCModel


@dataclass
class TwTi_RoRiAwAi(RCModel):
    """Envelope/indoor RC chain with separate envelope and indoor solar apertures."""

    states = (
        ("TEMPERATURE", "xw", "envelope temperature"),
        ("TEMPERATURE", "xi", "indoor temperature"),
    )

    params = (
        ("THERMAL_RESISTANCE", "Ro", "between the outdoor and the envelope"),
        ("THERMAL_RESISTANCE", "Ri", "between the envelope and the indoor"),
        ("THERMAL_CAPACITY", "Cw", "of the envelope"),
        ("THERMAL_CAPACITY", "Ci", "of the indoor"),
        ("SOLAR_APERTURE", "Aw", "of the envelope"),
        ("SOLAR_APERTURE", "Ai", "of the windows"),
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
        ("POWER", "Qgh", "solar irradiance"),
        ("POWER", "Qh", "HVAC system heat"),
    )

    outputs = (("TEMPERATURE", "xi", "indoor temperature"),)

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
            Aw,
            Ai,
            sigw_w,
            sigw_i,
            sigv,
            x0_w,
            x0_i,
            sigx0_w,
            sigx0_i,
            *_,
        ) = cast(Parameters, self.parameters).theta

        self.A = array_set(self.A, slice(None), [
            [-(Ro + Ri) / (Cw * Ri * Ro), 1.0 / (Cw * Ri)],
            [1.0 / (Ci * Ri), -1.0 / (Ci * Ri)],
        ])
        self.B = array_set(self.B, slice(None), [[1.0 / (Cw * Ro), Aw / Cw, 0.0], [0.0, Ai / Ci, 1.0 / Ci]])
        self.Q = array_set(self.Q, self._diag, [sigw_w, sigw_i])
        self.R = array_set(self.R, (0, 0), sigv)
        self.x0 = array_set(self.x0, (slice(None), 0), [x0_w, x0_i])
        self.P0 = array_set(self.P0, self._diag, [sigx0_w, sigx0_i])
