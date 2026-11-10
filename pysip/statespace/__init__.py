"""State-space model families and their dynamically populated model registry."""

from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from .base import StateSpace, model_registry
from .gaussian_process import (
    GPProduct,
    GPSum,
    Matern12,
    Matern32,
    Matern52,
    Periodic,
    QuasiPeriodic12,
    QuasiPeriodic32,
)
from .gaussian_process import __all__ as gaussian_process
from .thermal_network import (
    R2C2Qgh,
    Ti_RA,
    Ti_RAcv,
    TiTh_RwRhAi,
    TiTm_RwRmAi,
    TwTi_RoRi,
    TwTi_RoRiAwAi,
    TwTi_RoRiAwAicv,
    TwTi_RoRiRbAwAi,
    TwTi_RoRiRbAwAicv,
    TwTiTb_RoRiRbAwAicv,
    TwTiTb_RoRiRibRbbAwAicv,
    TwTiTh_RoRiRhAwAi,
    TwTiTh_RoRiRhAwAicv,
    TwTiTh_RoRiRhAwAicv_bis,
    TwTiTm_RoRiAwAi,
    TwTiTm_RoRiAwAicv,
    TwTiTm_RoRiRbAwAicv,
    TwTiTm_RoRiRmAwAi,
    TwTiTm_RoRiRmAwAiAm,
    TwTiTm_RoRiRmAwAicv,
    TwTiTm_RoRiRmRbAwAi,
    TwTiTm_RoRiRmRbAwAicv,
)
from .thermal_network import __all__ as thermal_network

package_dir = str(Path(__file__).resolve().parent)
for _, module_name, _ in iter_modules([package_dir]):
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute) and issubclass(attribute, StateSpace):
            globals()[attribute_name] = attribute

Models = model_registry

__all__ = [
    "GPProduct",
    "GPSum",
    "Matern12",
    "Matern32",
    "Matern52",
    "Models",
    "Periodic",
    "QuasiPeriodic12",
    "QuasiPeriodic32",
    "R2C2Qgh",
    "StateSpace",
    "TiTh_RwRhAi",
    "TiTm_RwRmAi",
    "Ti_RA",
    "Ti_RAcv",
    "TwTiTb_RoRiRbAwAicv",
    "TwTiTb_RoRiRibRbbAwAicv",
    "TwTiTh_RoRiRhAwAi",
    "TwTiTh_RoRiRhAwAicv",
    "TwTiTh_RoRiRhAwAicv_bis",
    "TwTiTm_RoRiAwAi",
    "TwTiTm_RoRiAwAicv",
    "TwTiTm_RoRiRbAwAicv",
    "TwTiTm_RoRiRmAwAi",
    "TwTiTm_RoRiRmAwAiAm",
    "TwTiTm_RoRiRmAwAicv",
    "TwTiTm_RoRiRmRbAwAi",
    "TwTiTm_RoRiRmRbAwAicv",
    "TwTi_RoRi",
    "TwTi_RoRiAwAi",
    "TwTi_RoRiAwAicv",
    "TwTi_RoRiRbAwAi",
    "TwTi_RoRiRbAwAicv",
    "gaussian_process",
    "thermal_network",
]
