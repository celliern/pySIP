"""Continuous-time stochastic state-space inference, simulation, and prediction."""

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from . import statespace as Models
from .regressors import Regressor
from .statespace_estimator import KalmanQR

__all__ = ["KalmanQR", "Models", "Regressor"]
