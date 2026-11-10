import numpy as np
import pandas as pd

from pysip import Regressor
from pysip.statespace import Matern32, Ti_RA


def test_single_free_parameter_fit_preserves_fixed_parameters():
    model = Matern32([
        {"name": "mscale", "value": 1.0, "transform": "fixed"},
        {"name": "lscale", "value": 0.8, "transform": "log"},
        {"name": "sigv", "value": 0.1, "transform": "fixed"},
    ])
    reg = Regressor(model, outputs="y")
    t = np.linspace(0, 1, 12)
    df = pd.DataFrame({"y": np.sin(12 * t)}, index=t)
    initial = reg.log_likelihood(df)
    reg.fit(df, disp=False)
    assert reg.log_likelihood(df) < initial
    np.testing.assert_array_equal(reg.parameters.theta[[0, 2]], [1.0, 0.1])


def test_prediction_and_output_estimation_share_positive_feedthrough_and_variance():
    values = {"R": 1.0, "C": 2.0, "A": 0.0, "sigw": 0.1, "sigv": 0.2, "x0": 1.0, "sigx0": 0.4}
    model = Ti_RA([
        {"name": name, "value": value, "transform": "fixed"} for name, value in values.items()
    ])
    model.D[0] = [2.0, -1.0, 0.5]
    reg = Regressor(model)
    df = pd.DataFrame({"To": [1.0, 2.0, 3.0], "Qgh": [3.0, 2.0, 1.0], "Qh": [2.0, 1.0, 0.0]})
    prediction = reg.predict(df, use_outputs=False)
    estimate = reg.estimate_output(df, use_outputs=False)
    expected_mean = prediction.x.values @ model.C.T + df[reg.inputs].to_numpy() @ model.D.T
    expected_variance = np.array([
        np.diag(model.C @ P @ model.C.T + model.R.T @ model.R)
        for P in prediction.P.values
    ])
    np.testing.assert_allclose(prediction.y_mean, expected_mean)
    np.testing.assert_allclose(prediction.y_std.values ** 2, expected_variance)
    np.testing.assert_allclose(estimate.y, expected_mean)
    np.testing.assert_allclose(estimate.y_std.values ** 2, expected_variance)
