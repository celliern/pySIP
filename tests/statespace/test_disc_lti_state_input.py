from time import time

import numpy as np
import pytest

from pysip.statespace import (
    TwTi_RoRiAwAicv,
    TwTiTh_RoRiRhAwAicv,
    TwTiTm_RoRiRmAwAicv,
    discretization,
)

sT = 3600.0 * 24.0

par_rc1 = [
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

par_rc2 = [
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

par_rc3 = [
    {"name": "Ro", "scale": 1e-2, "bounds": (1e-4, None)},
    {"name": "Ri", "scale": 1e-3, "bounds": (1e-4, None)},
    {"name": "Rh", "scale": 1e-3, "bounds": (1e-4, None)},
    {"name": "Cw", "scale": 1e7 / sT, "bounds": (1e-4, None)},
    {"name": "Ci", "scale": 1e6 / sT, "bounds": (1e-4, None)},
    {"name": "Ch", "scale": 1e6 / sT, "bounds": (1e-4, None)},
    {"name": "Aw", "scale": 1e0, "bounds": (1e-4, None)},
    {"name": "Ai", "scale": 1e0, "bounds": (1e-4, None)},
    {"name": "cv", "value": 5e2, "transform": "fixed"},
    {"name": "sigw_w", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigw_i", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigw_h", "scale": 1e-3 * sT**0.5, "bounds": (1e-4, None)},
    {"name": "sigv", "scale": 1e-2, "bounds": (1e-4, None)},
    {"name": "x0_w", "loc": 15.0, "scale": 7.0, "transform": "none"},
    {"name": "x0_i", "loc": 20.0, "scale": 7.0, "transform": "none"},
    {"name": "x0_h", "loc": 20.0, "scale": 7.0, "transform": "none"},
    {"name": "sigx0_w", "value": 0.1, "transform": "fixed"},
    {"name": "sigx0_i", "value": 0.1, "transform": "fixed"},
    {"name": "sigx0_h", "value": 0.1, "transform": "fixed"},
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
    TwTi_RoRiAwAicv(par_rc1),
    TwTiTm_RoRiRmAwAicv(par_rc2),
    TwTiTh_RoRiRhAwAicv(par_rc3),
]


@pytest.fixture
def random_dt():
    dt_list = [1.0 / 24.0, 1.0 / (2.0 * 24.0), 1.0 / (4.0 * 24.0), 1.0 / (6.0 * 24.0)]
    return np.random.choice(dt_list)


@pytest.mark.parametrize("model", model_list)
@pytest.mark.parametrize("order_hold", [0, 1])
def test_discretization_state_input(model, order_hold, random_dt):
    dt = random_dt
    model.parameters.eta = np.random.uniform(-1, 1, model.parameters.n_par)
    model.update()

    tic = time()
    Ad_expm, B0d_expm, B1d_expm = discretization.state_input(
        model.A, model.B, dt, order_hold, "expm"
    )
    toc = time() - tic
    print(f"expm: {toc:.4e}")

    tic = time()
    Ad, B0d, B1d = discretization.state_input(
        model.A, model.B, dt, order_hold, "analytic"
    )
    toc = time() - tic
    print(f"analytic: {toc:.4e}")

    assert np.allclose(Ad_expm, Ad)
    assert np.allclose(B0d_expm, B0d)
    assert np.allclose(B1d_expm, B1d)
