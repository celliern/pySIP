"""Square-root Kalman inference and simulation.

Kernel vectors are columns: states ``(nx, 1)``, inputs ``(nu, 1)`` and
observations ``(ny, 1)``; trajectories have a leading time axis. Kernel ``P``,
``P0``, ``Q``, ``R`` and innovation ``S`` are upper covariance factors:
the corresponding covariance is ``factor.T @ factor``. Returned
``KalmanResult.P`` instead contains full covariance matrices. NaN observations
are omitted from updates and retain NaN residual entries.
"""

import math
import warnings
from collections.abc import Hashable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple, overload

import numpy as np
import pandas as pd
import xarray as xr
from numba.core.errors import NumbaPerformanceWarning

from .statespace.base import StateSpace

#### Models, compatible with the Numba implementation of the Kalman filter ####


class States(NamedTuple):
    """System matrices; A/B0/B1/Q may have a final time axis.

    A/Q are ``(nx, nx)``, B0/B1 ``(nx, nu)``, C ``(ny, nx)``,
    D ``(ny, nu)`` and R ``(ny, ny)`` at one step. Q/R are upper factors.
    """

    A: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray


class KalmanResult(NamedTuple):
    """State means, full covariances, whitened residuals and upper factors.

    Fields x/P/k/S have shapes ``(nt, nx, 1)``, ``(nt, nx, nx)``,
    ``(nt, ny, 1)`` and ``(nt, ny, ny)``. k is not a Kalman gain;
    S is the innovation factor, not a covariance or marginal deviation.
    """

    x: np.ndarray
    P: np.ndarray
    k: np.ndarray
    S: np.ndarray

    def to_xarray(
        self, idx_name: Hashable, time: pd.Index | np.ndarray,
        states: Sequence[str], outputs: Sequence[str],
    ) -> xr.Dataset:
        """Label trajectory arrays, removing the trailing vector singleton axis."""
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "x": ((idx_name, "states"), self.x[:, :, 0]),
                "P": ((idx_name, "states", "states"), self.P),
                "k": ((idx_name, "outputs"), self.k[:, :, 0]),
                "S": ((idx_name, "outputs", "outputs"), self.S),
            },
            coords=coords,
        )
        return ds


def _allocate_kalman_res(n_timesteps: int, nx: int, ny: int) -> KalmanResult:
    """Allocate uninitialized time-major filtering result arrays."""
    return KalmanResult(
        np.empty((n_timesteps, nx, 1)),
        np.empty((n_timesteps, nx, nx)),
        np.empty((n_timesteps, ny, 1)),
        np.empty((n_timesteps, ny, ny)),
    )


def _pack_kalman_res(
    res: KalmanResult, i: int,
    value: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Write mean, full covariance, residual and factor into result row i."""
    (
        res.x[i],
        res.P[i],
        res.k[i],
        res.S[i],
    ) = value


class SimulationResults(NamedTuple):
    """Simulated outputs ``(nt, ny, 1)`` and states ``(nt, nx, 1)``."""

    y: np.ndarray
    x: np.ndarray

    def to_xarray(
        self, idx_name: Hashable, time: pd.Index | np.ndarray,
        states: Sequence[str], outputs: Sequence[str],
    ) -> xr.Dataset:
        """Label simulated trajectories and remove vector singleton axes."""
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "x": ((idx_name, "states"), self.x[:, :, 0]),
                "y": ((idx_name, "outputs"), self.y[:, :, 0]),
            },
            coords=coords,
        )
        return ds


def _allocate_simulation_res(n_timesteps: int, nx: int, ny: int) -> SimulationResults:
    """Allocate uninitialized time-major simulation arrays."""
    return SimulationResults(
        np.empty((n_timesteps, ny, 1)),
        np.empty((n_timesteps, nx, 1)),
    )


def _pack_simulate(
    res: SimulationResults, i: int, value: tuple[np.ndarray, np.ndarray],
) -> None:
    """Write an output/state pair into result row i in place."""
    res.y[i], res.x[i] = value


class OutputEstimateResult(NamedTuple):
    """Output means and marginal deviations, both shaped ``(nt, ny, 1)``."""

    y: np.ndarray
    y_std: np.ndarray

    def to_xarray(
        self, idx_name: Hashable, time: pd.Index | np.ndarray,
        states: Sequence[str], outputs: Sequence[str],
    ) -> xr.Dataset:
        """Label output means/deviations; states supplies the state coordinate."""
        coords = {
            idx_name: time,
            "states": states,
            "outputs": outputs,
        }
        ds = xr.Dataset(
            {
                "y": ((idx_name, "outputs"), self.y[:, :, 0]),
                "y_std": ((idx_name, "outputs"), self.y_std[:, :, 0]),
            },
            coords=coords,
        )
        return ds




#### Pure python / numpy implementation of the Kalman filter ####
# they are not efficient par their own, and are therefore jitted by numba


def _solve_tril_transpose_inplace(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A.T @ x = b`` for upper A, overwriting column vector b."""
    for i in range(b.shape[0]):
        for j in range(i):
            b[i, 0] -= A[j, i] * b[j, 0]
        b[i, 0] /= A[i, i]
    return b


def _update(
    C: np.ndarray, D: np.ndarray, R: np.ndarray, x: np.ndarray,
    P: np.ndarray, u: np.ndarray, y: np.ndarray, _Arru: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return updated mean/factor, whitened residual and innovation factor.

    Uses column vectors and upper covariance factors; overwrites the square
    ``(nx + ny, nx + ny)`` QR workspace _Arru, whose upper-right block is zero.
    """
    ny, _ = C.shape
    _Arru[:ny, :ny] = R
    _Arru[ny:, :ny] = P @ C.T
    _Arru[ny:, ny:] = P
    _, r_fact = np.linalg.qr(_Arru)
    S = r_fact[:ny, :ny]
    if ny == 1:
        k = (y - C @ x - D @ u) / S[0, 0]
        x = x + r_fact[:1, 1:].T * k
    else:
        k = _solve_tril_transpose_inplace(S, y - C @ x - D @ u)
        x = x + r_fact[:ny, ny:].T @ k
    P = r_fact[ny:, ny:]
    return x, P, k, S


def _predict(
    A: np.ndarray, B0: np.ndarray, B1: np.ndarray, Q: np.ndarray,
    x: np.ndarray, P: np.ndarray, u: np.ndarray, dtu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate column mean x and upper factor P without modifying inputs."""
    _, r = np.linalg.qr(np.vstack((P @ A.T, Q)))
    x = A @ x + B0 @ u + B1 @ dtu
    return x, r


def _update_observed(
    C: np.ndarray, D: np.ndarray, R: np.ndarray, x: np.ndarray,
    P: np.ndarray, u: np.ndarray, y: np.ndarray, _Arru: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update observed channels, mutating workspace and padding missing k/S with NaN."""
    ny, nx = C.shape
    if not np.isnan(y).any():
        return _update(C, D, R, x, P, u, y, _Arru)
    observed = np.flatnonzero(~np.isnan(y[:, 0]))
    k = np.full((ny, 1), np.nan)
    S = np.full((ny, ny), np.nan)
    if observed.size:
        # Selecting factor columns preserves the marginal covariance R.T @ R.
        _, R_obs = np.linalg.qr(R[:, observed])
        R_obs = np.ascontiguousarray(R_obs)
        work = np.zeros((nx + observed.size, nx + observed.size))
        x, P, k_obs, S_obs = _update(
            C[observed], D[observed], R_obs, x, P, u, y[observed], work
        )
        for i in range(observed.size):
            k[observed[i]] = k_obs[i]
            for j in range(observed.size):
                S[observed[i], observed[j]] = S_obs[i, j]
    return x, P, k, S


def _kalman_step(
    x: np.ndarray, P: np.ndarray, u: np.ndarray, dtu: np.ndarray,
    y: np.ndarray, states: States, _Arru: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update then predict; return updated x/P/k/S and predicted x/P.

    P and S are upper factors; only the shared QR workspace is overwritten.
    """
    x_up, P_up, k, S = _update_observed(
        states.C, states.D, states.R, x, P, u, y, _Arru
    )
    x_pred, P_pred = _predict(
        states.A, states.B0, states.B1, states.Q, x_up, P_up, u, dtu
    )
    return x_up, P_up, k, S, x_pred, P_pred


def _unpack_states(states: States, i: int) -> States:
    """Select time i from time-varying matrices, retaining static C/D/R."""
    return States(
        states.A[:, :, i],
        states.B0[:, :, i],
        states.B1[:, :, i],
        states.C,
        states.D,
        states.Q[:, :, i],
        states.R,
    )


def _log_likelihood(
    x0: np.ndarray, P0: np.ndarray, u: np.ndarray, dtu: np.ndarray,
    y: np.ndarray, states: States,
) -> float:
    """Sum negative log likelihood, omitting unobserved trailing forecasts."""
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    while n_timesteps and np.isnan(y[n_timesteps - 1]).all():
        n_timesteps -= 1
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    log_likelihood = 0.0
    for i in range(n_timesteps):
        y_i = y[i].reshape(-1, 1)
        u_i = u[i].reshape(-1, 1)
        dtu_i = dtu[i].reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x_up, P_up, k, S = _update_observed(
            states_i.C, states_i.D, states_i.R, x, P, u_i, y_i, _Arru
        )
        if i + 1 < n_timesteps:
            x, P = _predict(
                states_i.A, states_i.B0, states_i.B1, states_i.Q,
                x_up, P_up, u_i, dtu_i,
            )
        for j in range(ny):
            if not np.isnan(y_i[j, 0]):
                log_likelihood += (
                    0.5 * math.log(2.0 * math.pi)
                    + math.log(abs(S[j, j]))
                    + 0.5 * k[j, 0] ** 2
                )
    return log_likelihood


def _filtering(
    x0: np.ndarray, P0: np.ndarray, u: np.ndarray, dtu: np.ndarray,
    y: np.ndarray, states: States,
) -> KalmanResult:
    """Filter time-major u/dtu/y using column x0 and upper factor P0.

    Returns filtered means, full covariances, whitened residuals and upper
    innovation factors; fills newly allocated result/work arrays in place.
    """
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    res = _allocate_kalman_res(n_timesteps, nx, ny)
    for i in range(n_timesteps):
        y_i = y[i].reshape(-1, 1)
        u_i = u[i].reshape(-1, 1)
        dtu_i = dtu[i].reshape(-1, 1)
        states_i = _unpack_states(states, i)
        x_up, P_up, k, S, x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        _pack_kalman_res(res, i, (x_up, P_up.T @ P_up, k, S))
    return res


def _smoothing(
    x0: np.ndarray, P0: np.ndarray, u: np.ndarray, dtu: np.ndarray,
    y: np.ndarray, states: States,
) -> KalmanResult:
    """Apply RTS smoothing to filtered trajectories in newly allocated storage.

    Updates result x/P in place to smoothed means/full covariances; k/S retain
    filtering residuals/upper factors. Input shapes match _filtering.
    """
    x = x0
    P = P0
    n_timesteps = y.shape[0]
    ny, nx = states.C.shape
    dtype = states.A.dtype
    _Arru = np.zeros((nx + ny, nx + ny), dtype=dtype)
    xp = np.empty((n_timesteps, nx, 1), dtype=dtype)
    Pp = np.empty((n_timesteps, nx, nx), dtype=dtype)
    res = _allocate_kalman_res(n_timesteps, nx, ny)
    for i in range(n_timesteps):
        y_i = y[i].reshape(-1, 1)
        u_i = u[i].reshape(-1, 1)
        dtu_i = dtu[i].reshape(-1, 1)
        states_i = _unpack_states(states, i)
        xp[i] = x
        Pp[i] = P.T @ P
        x_up, P_up, k, S, x, P = _kalman_step(x, P, u_i, dtu_i, y_i, states_i, _Arru)
        _pack_kalman_res(res, i, (x_up, P_up.T @ P_up, k, S))

    for i in range(n_timesteps - 2, -1, -1):
        G = np.linalg.solve(Pp[i + 1], states.A[:, :, i] @ res.P[i]).T
        res.x[i, :, :] += G @ (res.x[i + 1, :, :] - xp[i + 1, :, :])
        res.P[i, :, :] += G @ (res.P[i + 1, :, :] - Pp[i + 1, :, :]) @ G.T
    return res


def _simulate(
    x0: np.ndarray, P0: np.ndarray, u: np.ndarray, dtu: np.ndarray,
    states: States, random_seed: int | None = None,
) -> SimulationResults:
    """Draw time-major state/output trajectories using upper noise factors.

    Seeds the kernel's NumPy random state when requested, including the initial
    draw from covariance P0.T @ P0; records outputs before each transition.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    ny, nx = states.C.shape
    x = x0 + P0.T @ np.random.randn(nx, 1)
    n_timesteps = u.shape[0]
    res = _allocate_simulation_res(n_timesteps, nx, ny)
    for i in range(n_timesteps):
        u_i = u[i].reshape(-1, 1)
        dtu_i = dtu[i].reshape(-1, 1)
        states_i = _unpack_states(states, i)
        y = states_i.C @ x + states_i.D @ u_i + states_i.R.T @ np.random.randn(ny, 1)
        _pack_simulate(res, i, (y, x))
        if i + 1 < n_timesteps:
            x = (
                states_i.A @ x
                + states_i.B0 @ u_i
                + states_i.B1 @ dtu_i
                + states_i.Q.T @ np.random.randn(nx, 1)
            )
    return res


# All above will be jitted by numba, if available. Otherwise, the pure python / numpy
# implementation will be used.
try:
    from numba import jit_module

    jit_module(nopython=True, nogil=True, cache=True)
except ImportError:
    warnings.warn("Numba not installed, using pure python implementation")


def output_moments(
    C: np.ndarray, D: np.ndarray, R: np.ndarray, x: np.ndarray,
    P: np.ndarray, u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return output mean and marginal standard deviation, shaped ``(..., ny, 1)``.

    C/D/R have shapes ``(ny, nx)``, ``(ny, nu)``, ``(ny, ny)``.
    x/u are column vectors with optional leading batch dimensions; P contains
    full state covariances ``(..., nx, nx)``, not factors. R is an upper
    measurement covariance factor. Inputs are not modified.
    """
    mean = C @ x + D @ u
    variance = np.sum((C @ P) * C, axis=-1) + np.sum(R * R, axis=0)
    return mean, np.sqrt(variance)[..., None]


def _estimate_output(
    x0: np.ndarray, P0: np.ndarray, u: np.ndarray, dtu: np.ndarray,
    y: np.ndarray, states: States,
) -> OutputEstimateResult:
    """Return filtered output means/deviations, including measurement noise."""
    res = _filtering(x0, P0, u, dtu, y, states)
    mean, std = output_moments(
        states.C, states.D, states.R, res.x, res.P, u[..., None]
    )
    return OutputEstimateResult(mean, std)


@dataclass
class KalmanQR:
    """Square-root Kalman inference for a mutable StateSpace model.

    Public trajectory methods update/discretize ss before running compiled
    kernels. Initial P0 and single-step P are upper covariance factors, while
    filtering/smoothing result P contains full covariances. Input pandas rows
    share observation times; u/dtu have nu columns and y has ny columns.
    """

    ss: StateSpace

    @overload
    def _proxy_params(
        self, dt: pd.Series, vars: tuple[pd.DataFrame, pd.DataFrame],
        x0: np.ndarray | None = None, P0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, States]:
        """Prepare simulation inputs without observations."""

    @overload
    def _proxy_params(
        self, dt: pd.Series, vars: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        x0: np.ndarray | None = None, P0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, States]:
        """Prepare inference inputs including observations."""

    def _proxy_params(
        self,
        dt: pd.Series,
        vars: Sequence[pd.DataFrame],
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> tuple[np.ndarray | States, ...]:
        """Update ss and prepare contiguous arrays plus time-indexed matrices.

        Returns x0, upper factor P0, each supplied time-major dataframe array,
        and States. Model matrices are refreshed in place.
        """
        ss = self.ss
        ss.update()
        dts, idx = np.unique(dt, return_inverse=True)
        # Column-major storage makes each time slice contiguous for LAPACK.
        A = np.empty((ss.nx, ss.nx, dt.size), order="F")
        B0 = np.empty((ss.nx, ss.nu, dt.size), order="F")
        B1 = np.empty((ss.nx, ss.nu, dt.size), order="F")
        Q = np.empty((ss.nx, ss.nx, dt.size), order="F")
        Ai, B0i, B1i, Qi = map(np.dstack, zip(*map(ss.discretization, dts)))
        A[:] = Ai[:, :, idx]
        B0[:] = B0i[:, :, idx]
        B1[:] = B1i[:, :, idx]
        Q[:] = Qi[:, :, idx]

        arrays = [np.ascontiguousarray(var.to_numpy()) for var in vars]
        states = States(A, B0, B1, ss.C, ss.D, Q, ss.R)
        x0 = np.ascontiguousarray(x0 if x0 is not None else ss.x0)
        # QR returns column-major factors; keep the loop's layout stable.
        P0 = np.asfortranarray(P0 if P0 is not None else ss.P0)
        return (x0, P0, *arrays, states)

    def update(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update a state without refreshing ss or modifying input arrays.

        Parameters
        ----------
        x : np.ndarray, shape (nx, 1)
            Prior state mean.
        P : np.ndarray, shape (nx, nx)
            Upper prior covariance factor.
        u : np.ndarray, shape (nu, 1)
            Exogenous inputs.
        y : np.ndarray, shape (ny, 1)
            Observations; NaN channels are omitted.

        Returns
        -------
        tuple of np.ndarray
            Updated mean, upper state factor, whitened residual k and upper
            innovation factor S. Missing-channel k/S entries are NaN.
            Call ss.update() explicitly after changing parameters.
        """
        x, P = deepcopy(x), deepcopy(P)
        nx = self.ss.nx
        ny = self.ss.ny
        C = self.ss.C
        D = self.ss.D
        R = self.ss.R
        _Arru = np.zeros((nx + ny, nx + ny))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _update_observed(C, D, R, x, P, u, y, _Arru)

    def predict(
        self,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        dtu: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict next mean and upper covariance factor without mutating inputs.

        Parameters
        ----------
        x : np.ndarray, shape (nx, 1)
            Current state mean.
        P : np.ndarray, shape (nx, nx)
            Upper state covariance factor.
        u : np.ndarray, shape (nu, 1)
            Exogenous inputs.
        dtu : np.ndarray, shape (nu, 1)
            Input time derivatives.
        dt : float
            Transition duration.

        Returns
        -------
        tuple of np.ndarray
            Predicted column mean and upper factor of the next covariance.
        """
        x, P = deepcopy(x), deepcopy(P)
        A, B0, B1, Q = self.ss.discretization(dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            return _predict(A, B0, B1, Q, x, P, u, dtu)

    def log_likelihood(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
    ) -> float:
        """Return scalar negative Gaussian log likelihood of observed data.

        Parameters
        ----------
        dt : pd.Series, shape (nt,)
            Time steps in model units.
        u : pd.DataFrame, shape (nt, nu)
            Exogenous inputs.
        dtu : pd.DataFrame, shape (nt, nu)
            Input time derivatives.
        y : pd.DataFrame, shape (nt, ny)
            Observations; NaNs are omitted.

        Returns
        -------
        float
            Summed negative log likelihood, not pointwise densities.
            Refreshes model matrices using current parameters.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u_array, dtu_array, y_array, states = self._proxy_params(dt, (u, dtu, y))

            return _log_likelihood(x0, P0, u_array, dtu_array, y_array, states)

    def filtering(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> KalmanResult:
        """Return filtered states and innovations, refreshing model matrices.

        Parameters
        ----------
        dt : pd.Series, shape (nt,)
            Time steps in model units.
        u : pd.DataFrame, shape (nt, nu)
            Exogenous inputs.
        dtu : pd.DataFrame, shape (nt, nu)
            Input time derivatives.
        y : pd.DataFrame, shape (nt, ny)
            Observations; NaN channels are omitted from each update.
        x0 : np.ndarray, shape (nx, 1), optional
            Initial mean; defaults to ss.x0.
        P0 : np.ndarray, shape (nx, nx), optional
            Upper initial covariance factor; defaults to ss.P0.

        Returns
        -------
        KalmanResult
            x: filtered means ``(nt, nx, 1)``; P: full covariances
            ``(nt, nx, nx)``; k: whitened residuals ``(nt, ny, 1)``;
            S: upper innovation factors ``(nt, ny, ny)``.
            Missing-channel k/S entries are NaN. Input arrays are not modified.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u_array, dtu_array, y_array, states = self._proxy_params(
                dt, (u, dtu, y), x0, P0
            )

            return _filtering(x0, P0, u_array, dtu_array, y_array, states)

    def smoothing(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> KalmanResult:
        """Return RTS-smoothed states with filtering innovations.

        Parameters
        ----------
        dt : pd.Series, shape (nt,)
            Time steps in model units.
        u : pd.DataFrame, shape (nt, nu)
            Exogenous inputs.
        dtu : pd.DataFrame, shape (nt, nu)
            Input time derivatives.
        y : pd.DataFrame, shape (nt, ny)
            Observations; NaN channels are omitted.
        x0 : np.ndarray, shape (nx, 1), optional
            Initial mean; defaults to ss.x0.
        P0 : np.ndarray, shape (nx, nx), optional
            Upper initial covariance factor; defaults to ss.P0.

        Returns
        -------
        KalmanResult
            Smoothed means x and full covariances P; k and S remain the
            filtering whitened residuals and upper innovation factors.
            Shapes match filtering(). Refreshes model matrices and does not
            modify input arrays.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u_array, dtu_array, y_array, states = self._proxy_params(
                dt, (u, dtu, y), x0, P0
            )

            return _smoothing(x0, P0, u_array, dtu_array, y_array, states)

    def simulate(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        *,
        random_seed: int | None = None,
    ) -> SimulationResults:
        r"""Simulate states and outputs at the same observation times.

        The initial state has covariance ``P0.T @ P0``. Measurement and process
        noise have covariances ``R.T @ R`` and ``Q.T @ Q``, respectively.
        The output is ``C @ x + D @ u`` plus measurement noise; the transition
        occurs after recording the current state and output.

        Parameters
        ----------
        dt : pd.Series
            Time steps.
        u : pd.DataFrame
            Exogenous inputs, shape (nt, nu).
        dtu : pd.DataFrame
            Input time derivatives, shape (nt, nu).
        x0 : np.ndarray, optional
            Initial column mean (nx, 1); defaults to ss.x0.
        P0 : np.ndarray, optional
            Upper initial covariance factor (nx, nx); defaults to ss.P0.
        random_seed : int, optional
            Seed for reproducible simulation, including the initial state draw.

        Returns
        -------
        SimulationResults
            Simulated data in the form of a namedtuple with the following fields:

            - x: np.ndarray[n_timesteps, nx, 1]
                State vector.
            - y: np.ndarray[n_timesteps, ny, 1]
                Output vector.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u_array, dtu_array, states = self._proxy_params(dt, (u, dtu), x0, P0)
            return _simulate(x0, P0, u_array, dtu_array, states, random_seed)

    def estimate_output(
        self,
        dt: pd.Series,
        u: pd.DataFrame,
        dtu: pd.DataFrame,
        y: pd.DataFrame,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> OutputEstimateResult:
        """Return filtered output means and marginal standard deviations.

        Parameters
        ----------
        dt : pd.Series, shape (nt,)
            Time steps in model units.
        u : pd.DataFrame, shape (nt, nu)
            Exogenous inputs.
        dtu : pd.DataFrame, shape (nt, nu)
            Input time derivatives.
        y : pd.DataFrame, shape (nt, ny)
            Observations; NaN channels are omitted.
        x0 : np.ndarray, shape (nx, 1), optional
            Initial mean; defaults to ss.x0.
        P0 : np.ndarray, shape (nx, nx), optional
            Upper initial covariance factor; defaults to ss.P0.

        Returns
        -------
        OutputEstimateResult
            y and y_std, both ``(nt, ny, 1)``, include measurement noise in
            the uncertainty. Refreshes model matrices without modifying inputs.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
            x0, P0, u_array, dtu_array, y_array, states = self._proxy_params(
                dt, (u, dtu, y), x0, P0
            )

            return _estimate_output(x0, P0, u_array, dtu_array, y_array, states)
