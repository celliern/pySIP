import numpy as np
import pandas as pd
import pytest

from pysip.statespace import LatentForceModel, Matern32, R2C2_Qgh_Matern32, R2C2Qgh
from pysip.utils.artificial_data import generate_random_binary, generate_sine

sT = 24.0 * 60.0
sC = 1e8 / sT


@pytest.fixture
def artificial_data():
    """Generate 50 data points spaced of 1 hour

    Ti: Indoor temperature
    To: Outdoor temperature
    Tb: Boundary temperature
    Qgh: Global horizontal solar radiation
    Qh: Heat supplied by heaters
    Qv: Heat supplied by ventilation
    """
    n = 50  # number of points
    df = pd.DataFrame(
        {
            "Ti": generate_sine(n=n, amplitude=3.0, phase=np.pi / 3.0, offset=20.0)[1],
            "To": generate_sine(n=n)[1],
            "Tb": generate_sine(n=n, amplitude=2.0, phase=np.pi / 4.0, offset=14.0)[1],
            "Qgh": generate_sine(n=n, amplitude=800.0, clip_to_0=True)[1],
            "Qh": generate_random_binary(n=n),
            "Qv": generate_sine(n=n, amplitude=200.0, offset=100.0)[1],
        }
    )
    df.index /= sT
    return df


@pytest.fixture
def models():
    p = np.random.uniform(-1, 1, 13)

    p_truth = [
        {"name": "R2C2Qgh__Ro", "scale": 1e-2, "transform": "log"},
        {"name": "R2C2Qgh__Ri", "scale": 1e-3, "transform": "log"},
        {"name": "R2C2Qgh__Cw", "scale": sC, "transform": "log"},
        {"name": "R2C2Qgh__Ci", "scale": sC, "transform": "log"},
        {"name": "R2C2Qgh__sigw_w", "scale": np.sqrt(sT), "transform": "log"},
        {"name": "R2C2Qgh__sigw_i", "scale": np.sqrt(sT), "transform": "log"},
        {"name": "R2C2Qgh__sigv", "scale": 1e-2, "transform": "log"},
        {"name": "R2C2Qgh__x0_w", "scale": 1e1, "transform": "log"},
        {"name": "R2C2Qgh__x0_i", "scale": 1e1, "transform": "log"},
        {"name": "R2C2Qgh__sigx0_w", "scale": 1.0, "transform": "log"},
        {"name": "R2C2Qgh__sigx0_i", "scale": 1.0, "transform": "log"},
        {"name": "Matern32__mscale", "scale": 1e3, "transform": "log"},
        {"name": "Matern32__lscale", "scale": 1.0, "transform": "log"},
    ]

    p_rc = [
        {"name": "Ro", "scale": 1e-2, "transform": "log"},
        {"name": "Ri", "scale": 1e-3, "transform": "log"},
        {"name": "Cw", "scale": sC, "transform": "log"},
        {"name": "Ci", "scale": sC, "transform": "log"},
        {"name": "sigw_w", "scale": np.sqrt(sT), "transform": "log"},
        {"name": "sigw_i", "scale": np.sqrt(sT), "transform": "log"},
        {"name": "sigv", "scale": 1e-2, "transform": "log"},
        {"name": "x0_w", "scale": 1e1, "transform": "log"},
        {"name": "x0_i", "scale": 1e1, "transform": "log"},
        {"name": "sigx0_w", "scale": 1.0, "transform": "log"},
        {"name": "sigx0_i", "scale": 1.0, "transform": "log"},
    ]

    p_gp = [
        {"name": "mscale", "scale": 1e3, "transform": "log"},
        {"name": "lscale", "scale": 1.0, "transform": "log"},
        {"name": "sigv", "value": 0.0, "transform": "fixed"},
    ]

    truth = R2C2_Qgh_Matern32(p_truth)
    truth.parameters.eta_free = p
    truth.update()

    lfm = LatentForceModel(R2C2Qgh(p_rc), Matern32(p_gp), "Qgh")
    lfm.parameters.eta_free = p
    lfm.update()

    return truth, lfm


@pytest.fixture
def models_by_operator():
    truth = R2C2_Qgh_Matern32()
    theta = np.random.random(len(truth.parameters.ids))

    truth.parameters.theta_free = theta
    truth.update()

    lfm = R2C2Qgh(latent_forces="Qgh") <= Matern32()
    lfm.parameters.theta_free = theta
    lfm.update()
    return truth, lfm


def test_LFM_creation_by_function(models):
    LFM_creation(models)


def test_LFM_creation_by_operator(models_by_operator):
    LFM_creation(models_by_operator)


def LFM_creation(models):
    truth, lfm = models
    assert truth.states == lfm.states
    assert truth.inputs == lfm.inputs
    assert truth.outputs == lfm.outputs
    assert np.allclose(truth.parameters.theta_free, lfm.parameters.theta_free)

    assert np.allclose(truth.A, lfm.A)
    assert np.allclose(truth.B, lfm.B)
    assert np.allclose(truth.C, lfm.C)
    assert np.allclose(truth.D, lfm.D)
    assert np.allclose(truth.Q, lfm.Q)
    assert np.allclose(truth.R, lfm.R)
    assert np.allclose(truth.x0, lfm.x0)
    assert np.allclose(truth.P0, lfm.P0)


@pytest.mark.parametrize("latent_force", [[], ["Qgh", "Qh"], ["Qgh", "Qgh"]])
def test_rejects_unsupported_latent_force_count(latent_force):
    with pytest.raises(ValueError, match="Exactly one latent-force input"):
        LatentForceModel(R2C2Qgh(), Matern32(), latent_force)


def test_single_input_list_matches_string():
    rc, gp = R2C2Qgh(), Matern32()
    by_string = LatentForceModel(rc, gp, "Qgh")
    by_list = LatentForceModel(rc, gp, ["Qgh"])
    for model in (by_string, by_list):
        model.parameters.theta_free = np.ones(model.parameters.n_par_free)
        model.update()
    np.testing.assert_allclose(by_string.A, by_list.A)
    assert by_string.inputs == by_list.inputs
