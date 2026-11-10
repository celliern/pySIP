import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import xarray as xr
from scipy.optimize import approx_fprime

from pysip.params.prior import Gamma
from pysip.regressors import Regressor, _Loglike
from pysip.statespace import Matern32
from pysip.utils.save import load_model, save_model


def test_bayesian_inference_predictive_and_serialization(tmp_path):
    df = pd.DataFrame({"y": [0.1, 0.2, -0.1, 0.0]}, index=[0.0, 0.2, 0.4, 0.6])
    reg = Regressor(
        Matern32(
            [
                {"name": "mscale", "value": 1.0, "scale": 2.0, "transform": "log", "prior": Gamma(3, 3)},
                {"name": "lscale", "value": 0.5, "scale": 0.5, "transform": "log", "prior": Gamma(3, 6)},
                {"name": "sigv", "value": 0.1, "scale": 0.2, "transform": "log", "prior": Gamma(3, 30)},
            ]
        ),
        outputs="y",
    )
    model = reg.pymc_model(df)
    point = model.initial_point()
    names = list(point)
    values = np.array([point[name] for name in names])
    logp = model.compile_logp()
    gradient = model.compile_dlogp()(point)
    numerical = approx_fprime(
        values,
        lambda x: logp(dict(zip(names, x))),
        epsilon=1e-5,
    )
    np.testing.assert_allclose(gradient, numerical, rtol=1e-3, atol=1e-4)
    standardized = np.array([1.0, 0.5, 0.1])
    reg.parameters.theta_sd = standardized
    np.testing.assert_allclose(
        _Loglike(reg, df)(pt.as_tensor_variable(standardized)).eval(), reg.log_likelihood(df)
    )

    prior_prediction = reg.prior_predictive(df, samples=2, random_seed=123)
    assert prior_prediction["y"].shape == (2, len(df))
    altered = df.assign(y=1000.0)
    independent = reg.prior_predictive(altered, samples=2, random_seed=123)
    xr.testing.assert_identical(prior_prediction, independent)
    prior = reg.trace["prior"].to_dataset()
    reg.parameters.theta_sd = [
        prior[name].isel(chain=0, draw=0).item() for name in reg.parameters.ids_free
    ]
    first_seed = int(np.random.default_rng(123).integers(0, 2**31))
    expected = reg.estimator.simulate(
        *reg.prepare_data(df, with_outputs=False)[:3], random_seed=first_seed
    )
    np.testing.assert_allclose(prior_prediction["y"].isel(draw=0), expected.y[:, 0, 0])

    trace = reg.sample(
        df,
        draws=8,
        tune=8,
        chains=2,
        cores=1,
        random_seed=123,
        progressbar=False,
        compute_convergence_checks=False,
    )
    assert isinstance(trace, xr.DataTree)
    assert trace["posterior"].sizes["draw"] == 8
    assert trace["prior"].sizes["draw"] == 2
    assert reg.trace is trace
    assert set(az.summary(trace).index) == set(reg.parameters.ids_free)
    az.plot_trace(trace, backend="matplotlib")
    plt.close("all")

    reg._trace = trace.isel(draw=slice(0, 2))
    posterior_prediction = reg.posterior_predictive(df, random_seed=456)
    assert posterior_prediction["y"].shape == (2, 2, len(df))
    posterior = reg.trace["posterior"].to_dataset()
    reg.parameters.theta_sd = [
        posterior[name].isel(chain=0, draw=0).item() for name in reg.parameters.ids_free
    ]
    first_seed = int(np.random.default_rng(456).integers(0, 2**31))
    expected = reg.estimator.simulate(
        *reg.prepare_data(df, with_outputs=False)[:3], random_seed=first_seed
    )
    np.testing.assert_allclose(
        posterior_prediction["y"].isel(chain=0, draw=0), expected.y[:, 0, 0]
    )
    save_model(tmp_path / "regressor", reg)
    restored = load_model(tmp_path / "regressor")
    xr.testing.assert_identical(restored.trace, reg.trace)



def test_bayesian_composed_kernel_identifiers():
    def kernel():
        return Matern32([
            {"name": "mscale", "value": 1.0, "transform": "log", "prior": Gamma(3, 3)},
            {"name": "lscale", "value": 0.5, "transform": "log", "prior": Gamma(3, 6)},
            {"name": "sigv", "value": 0.1, "transform": "log", "prior": Gamma(3, 30)},
        ])

    reg = Regressor(kernel() + kernel(), outputs="y")
    df = pd.DataFrame({"y": [0.1, 0.2]}, index=[0.0, 0.2])
    model = reg.pymc_model(df)
    assert set(model.named_vars) == {*reg.parameters.ids_free, "likelihood"}
    draws = reg.prior_predictive(df, samples=2, random_seed=12)
    assert np.isfinite(draws["y"]).all()
