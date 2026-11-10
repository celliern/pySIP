import numpy as np
import pytest

from pysip.statespace import GPModel, Models, StateSpace, discretization

sT = 3600.0 * 24.0

par_rc2 = [
    {"name": "Ro", "scale": 1e-2, "bounds": (1e-4, None)},
    {"name": "Ri", "scale": 1e-3, "bounds": (1e-4, None)},
    {"name": "Cw", "scale": 1e7 / sT, "bounds": (1e-4, None)},
    {"name": "Ci", "scale": 1e6 / sT, "bounds": (1e-4, None)},
    {"name": "Aw", "scale": 1e0, "bounds": (1e-4, None)},
    {"name": "Ai", "scale": 1e0, "bounds": (1e-4, None)},
    {"name": "cv", "value": 5e2, "transform": "fixed"},
    {"name": "sigw_w", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigw_i", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigv", "scale": 1e-2, "bounds": (1e-4, None)},
    {"name": "x0_w", "loc": 15.0, "scale": 7.0, "transform": "none"},
    {"name": "x0_i", "loc": 20.0, "scale": 7.0, "transform": "none"},
    {"name": "sigx0_w", "value": 0.1, "transform": "fixed"},
    {"name": "sigx0_i", "value": 0.1, "transform": "fixed"},
]

par_rc3 = [
    {"name": "Ro", "scale": 1e-2, "bounds": (1e-4, None)},
    {"name": "Ri", "scale": 1e-3, "bounds": (1e-4, None)},
    {"name": "Rm", "scale": 1e-3, "bounds": (1e-4, None)},
    {"name": "Cw", "scale": 1e7 / sT, "bounds": (1e-4, None)},
    {"name": "Ci", "scale": 1e6 / sT, "bounds": (1e-4, None)},
    {"name": "Cm", "scale": 1e7 / sT, "bounds": (1e-4, None)},
    {"name": "Aw", "scale": 1e0, "bounds": (1e-4, None)},
    {"name": "Ai", "scale": 1e0, "bounds": (1e-4, None)},
    {"name": "cv", "value": 5e2, "transform": "fixed"},
    {"name": "sigw_w", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigw_i", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigw_m", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigv", "scale": 1e-2, "bounds": (1e-4, None)},
    {"name": "x0_w", "loc": 15.0, "scale": 7.0, "transform": "none"},
    {"name": "x0_i", "loc": 20.0, "scale": 7.0, "transform": "none"},
    {"name": "x0_m", "loc": 20.0, "scale": 7.0, "transform": "none"},
    {"name": "sigx0_w", "value": 0.1, "transform": "fixed"},
    {"name": "sigx0_i", "value": 0.1, "transform": "fixed"},
    {"name": "sigx0_m", "value": 0.1, "transform": "fixed"},
]

par_p = [
    {"name": "period", "bounds": (1e-4, None)},
    {"name": "mscale", "bounds": (1e-4, None)},
    {"name": "lscale", "bounds": (0.05, None)},
    {"name": "sigv", "bounds": (1e-4, None)},
]

par_m = [
    {"name": "mscale", "bounds": (1e-4, None)},
    {"name": "lscale", "bounds": (1e-4, None)},
    {"name": "sigv", "bounds": (1e-4, None)},
]

par_qp = [
    {"name": "period", "bounds": (1e-4, None)},
    {"name": "mscale", "bounds": (1e-4, None)},
    {"name": "lscale", "bounds": (0.05, None)},
    {"name": "sigv", "bounds": (1e-4, None)},
    {"name": "decay", "bounds": (1e-4, None)},
]

model_list = [
    Models.TwTi_RoRiAwAicv(par_rc2),
    Models.TwTiTm_RoRiRmAwAicv(par_rc3),
    Models.Matern12(parameters=par_m),
    Models.Matern32(parameters=par_m),
    Models.Matern52(parameters=par_m),
    Models.Periodic(parameters=par_p) + Models.Matern12(parameters=par_m),
    Models.Periodic(parameters=par_p) + Models.Matern32(parameters=par_m),
    Models.Periodic(parameters=par_p) + Models.Matern52(parameters=par_m),
    Models.QuasiPeriodic12(parameters=par_qp),
    Models.QuasiPeriodic32(parameters=par_qp),
    Models.Periodic(parameters=par_p) * Models.Matern52(parameters=par_m),
    Models.QuasiPeriodic12(parameters=par_qp) + Models.Matern12(parameters=par_m),
    Models.QuasiPeriodic32(parameters=par_qp) + Models.Matern12(parameters=par_m),
    Models.QuasiPeriodic32(parameters=par_qp) + Models.Matern32(parameters=par_m),
    Models.QuasiPeriodic32(parameters=par_qp) + Models.Matern32(parameters=par_m),
    Models.QuasiPeriodic12(parameters=par_qp) + Models.Matern52(parameters=par_m),
    Models.QuasiPeriodic32(parameters=par_qp) + Models.Matern52(parameters=par_m),
    Models.TwTi_RoRiAwAicv(par_rc2, latent_forces="Qv")
    <= Models.Matern32(parameters=par_m),
    Models.TwTi_RoRiAwAicv(par_rc2, latent_forces="Qv")
    <= Models.Periodic(parameters=par_p),
    Models.TwTi_RoRiAwAicv(par_rc2, latent_forces="Qv")
    <= Models.QuasiPeriodic32(parameters=par_qp),
    Models.TwTiTm_RoRiRmAwAicv(par_rc3, latent_forces="Qv")
    <= Models.Matern32(parameters=par_m),
    Models.TwTiTm_RoRiRmAwAicv(par_rc3, latent_forces="Qv")
    <= Models.Periodic(parameters=par_p),
    Models.TwTiTm_RoRiRmAwAicv(par_rc3, latent_forces="Qv")
    <= Models.QuasiPeriodic32(parameters=par_qp),
]


@pytest.fixture
def random_dt():
    dt_list = [1.0 / 24.0, 1.0 / (2.0 * 24.0), 1.0 / (4.0 * 24.0), 1.0 / (6.0 * 24.0)]
    return np.random.choice(dt_list)


@pytest.mark.parametrize("model", model_list)
def test_disc_LTI(model: StateSpace, random_dt):
    dt = random_dt

    model.parameters.eta = np.random.uniform(-1, 1, model.parameters.n_par)
    ssm = model.get_discrete_ssm(dt)

    QQ = model.Q.T @ model.Q

    Qd = ssm.Q.T @ ssm.Q

    Qd_mfd = discretization.diffusion_mfd(model.A, QQ, dt)
    Qd_lyap = discretization.diffusion_lyap(model.A, QQ, ssm.A)
    Qd_kron = discretization.diffusion_kron(model.A, QQ, ssm.A)


    assert np.allclose(Qd_kron, Qd)
    assert np.allclose(Qd_lyap, Qd)
    assert np.allclose(Qd_mfd, Qd)

    if isinstance(model, GPModel):
        Pinf = model.P0.T @ model.P0
        Qd_statio = discretization.diffusion_stationary(Pinf, ssm.A)

        assert np.allclose(Qd_statio, Qd)
