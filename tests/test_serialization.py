import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pysip import __version__
from pysip.regressors import Regressor
from pysip.statespace import Matern32
from pysip.utils.save import load_model, save_model


def test_regressor_serialization_roundtrip(tmp_path):
    reg = Regressor(
        Matern32(
            [
                {"name": "mscale", "value": 1.0, "transform": "log"},
                {"name": "lscale", "value": 0.5, "transform": "log"},
                {"name": "sigv", "value": 0.1, "transform": "log"},
            ]
        ),
        outputs="y",
    )
    data = pd.DataFrame({"y": [0.1, 0.2, -0.1, 0.0]}, index=[0.0, 0.2, 0.4, 0.6])
    expected = reg.estimate_output(data)
    expected_likelihood = reg.log_likelihood(data)
    parameters = reg.parameters.theta.copy()
    path = tmp_path / "regressor"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_model(path, reg)
        restored = load_model(path)
    assert restored is not reg
    assert restored.__version__ == __version__
    np.testing.assert_array_equal(reg.parameters.theta, parameters)
    np.testing.assert_array_equal(restored.parameters.theta, parameters)
    assert restored.estimator.ss is restored.ss
    xr.testing.assert_identical(restored.estimate_output(data), expected)
    assert restored.log_likelihood(data) == pytest.approx(expected_likelihood)
    # Saving must not disable or otherwise mutate the live estimator.
    xr.testing.assert_identical(reg.estimate_output(data), expected)


def test_serialization_propagates_real_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing")
    invalid = tmp_path / "invalid.pickle"
    invalid.write_bytes(b"not a pickle")
    with pytest.raises(pickle.UnpicklingError):
        load_model(invalid)
    with pytest.raises((AttributeError, pickle.PicklingError)):
        save_model(tmp_path / "unpicklable", SimpleNamespace(callback=lambda: None))
