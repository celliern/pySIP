"""Exact LTI discretization without mutating input arrays.

All diffusion routines accept full continuous covariance rates, not upper
factors, and return full discrete covariances. Zero covariance inputs may be
returned by identity. Time units must agree with the continuous matrices.
"""


import numpy as np
from numpy.linalg import LinAlgError, lstsq, solve
from scipy.linalg import (
    LinAlgWarning,
    expm,
    solve_continuous_lyapunov,
)


def state_input(
    A: np.ndarray,
    B: np.ndarray,
    dt: float = 1.0,
    order_hold: int = 0,
    method: str = "expm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the state and input matrices

    $$
    x_{k+1} = Ad x_k + B0d u_k + B1d \\dot{u}_k
    $$

    Parameters
    ----------
    A : ndarray
        Continuous state matrix, shape (nx, nx).
    B : ndarray
        Continuous input matrix, shape (nx, nu).
    dt : float
        Sampling time
    order_hold : int
        zero order hold = 0 or first order hold = 1
    method : str
        Augmented exponential ``expm`` or linear-solve ``analytic``.
        The analytic method requires nonsingular A.

    Returns
    -------
    Ad : ndarray
        Discrete state matrix, shape (nx, nx).
    B0d : ndarray
        Input-value matrix, shape (nx, nu).
    B1d : ndarray
        Input-derivative matrix, shape (nx, nu); zero for zero-order hold.

    Raises
    ------
    ValueError
        The method is neither ``expm`` nor ``analytic``.
    numpy.linalg.LinAlgError
        A is singular when the analytic method is selected.
    """
    if method == "expm":
        return state_input_expm(A, B, dt, order_hold)
    elif method == "analytic":
        Ad = state(A, dt)
        B0d, B1d = input_analytic(A, B, Ad, dt, order_hold)
        return Ad, B0d, B1d
    else:
        raise ValueError("`method must be `expm` or `analytic`")


def state(A: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Discretize the state matrix

    Parameters
    ----------
    A : array_like
        Continuous state matrix, shape (nx, nx); not mutated.
    dt : float
        Sampling time

    Returns
    -------
    Ad : array_like
        Discrete state matrix, shape (nx, nx).
    """
    return expm(A * dt)


def input_analytic(
    A: np.ndarray, B: np.ndarray, Ad: np.ndarray, dt: float = 1.0, order_hold: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize the input matrix with the analytic expression

    Parameters
    ----------
    A : array_like
        Nonsingular continuous state matrix, shape (nx, nx); not mutated.
    B : array_like
        Continuous input matrix, shape (nx, nu); not mutated.
    Ad : array_like
        Discrete state matrix, shape (nx, nx); not mutated.
    dt : float
        Sampling time
    order_hold : int
        zero order hold = 0 or first order hold = 1

    Returns
    -------
    B0d : array_like
        Input-value matrix, shape (nx, nu).
    B1d : array_like
        Input-derivative matrix, shape (nx, nu); zero for zero-order hold.

    Raises
    ------
    numpy.linalg.LinAlgError
        The continuous state matrix is singular.

    Notes
    -----
    This method requires a nonsingular state matrix. Use ``state_input`` with
    ``method="expm"`` for singular matrices.
    """
    nx, nu = B.shape
    B0d = solve(A, (Ad - np.eye(nx)) @ B)
    if order_hold == 0:
        B1d = np.zeros((nx, nu))
    else:
        B1d = solve(A, B0d - dt * B)

    return B0d, B1d


def state_input_expm(
    A: np.ndarray, B: np.ndarray, dt: float = 1.0, order_hold: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the state and input matrices with matrix exponential

    Parameters
    ----------
    A : numpy.array
        Continuous state matrix, shape (nx, nx); may be singular; not mutated.
    B : numpy.array
        Continuous input matrix, shape (nx, nu); not mutated.
    dt : float
        Sampling time
    order_hold : int
        zero order hold = 0 or first order hold = 1

    Returns
    -------
    Ad : numpy.array
        Discrete state matrix, shape (nx, nx).
    B0d : numpy.array
        Input-value matrix, shape (nx, nu).
    B1d : numpy.array
        Input-derivative matrix, shape (nx, nu); zero for zero-order hold.
    """
    nx, nu = B.shape
    if order_hold == 0:
        F = np.zeros((nx + nu, nx + nu), dtype=A.dtype)
        F[:nx, :nx] = A
        F[:nx, nx:] = B
        Ad, B0d = np.split(expm(F * dt)[:nx, :], indices_or_sections=[nx], axis=1)
        B1d = np.zeros((nx, nu))
    else:
        F = np.zeros((nx + 2 * nu, nx + 2 * nu), dtype=A.dtype)
        F[:nx, :nx] = A
        F[:nx, nx : nx + nu] = B
        F[nx : nx + nu, nx + nu :] = np.eye(nu)
        Ad, B0d, B1d = np.split(
            expm(F * dt)[:nx, :], indices_or_sections=[nx, nx + nu], axis=1
        )
    return Ad, B0d, B1d


def diffusion_mfd(A: np.ndarray, Q: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Discretize the diffusion matrix by Matrix Fraction Decomposition.

    A short-step Van Loan exponential followed by covariance doubling avoids
    overflow in the unstable ``-A.T`` block for long sampling intervals.

    Parameters
    ----------
    A : array_like
        Continuous state matrix, shape (nx, nx).
    Q : array_like
        Full continuous diffusion covariance rate, shape (nx, nx), not a factor.
    dt : float
        Sampling time.

    Returns
    -------
    Qd : ndarray
        Full discrete process covariance, shape (nx, nx).
    """
    if not Q.any():
        return Q

    nx = A.shape[0]
    F = np.zeros((2 * nx, 2 * nx), dtype=A.dtype)
    F[:nx, :nx] = A
    F[nx:, nx:] = -A.T
    F[:nx, nx:] = Q
    magnitude = np.linalg.norm(A, ord=np.inf) * abs(dt)
    squarings = max(0, int(np.ceil(np.log2(magnitude)))) if magnitude > 1.0 else 0
    Fd = expm(F * (dt / 2.0**squarings))[:nx, :]
    Ad = Fd[:, :nx]
    Qd = Fd[:, nx:] @ Ad.T
    for _ in range(squarings):
        Qd = Qd + Ad @ Qd @ Ad.T
        Ad = Ad @ Ad
    return (Qd + Qd.T) / 2.0


def diffusion_lyap(A: np.ndarray, Q: np.ndarray, Ad: np.ndarray) -> np.ndarray:
    """Discretize the diffusion matrix by solving the Lyapunov equation

    Parameters
    ----------
    A : array_like
        Continuous state matrix, shape (nx, nx).
    Q : array_like
        Full continuous diffusion covariance rate, shape (nx, nx), not a factor.
    Ad : array_like
        Discrete state matrix, shape (nx, nx).

    Returns
    -------
    Qd : array_like
        Full discrete process covariance, shape (nx, nx).
    """
    if not Q.any():
        return Q
    return solve_continuous_lyapunov(A, -Q + Ad @ Q @ Ad.T)


def diffusion_stationary(Pinf: np.ndarray, Ad: np.ndarray) -> np.ndarray:
    """
    Discretize the stationary covariance matrix

    Parameters
    ----------
    Pinf : numpy.array
        Full stationary covariance, shape (nx, nx), not an upper factor.
    Ad : numpy.array
        Discrete state matrix, shape (nx, nx).

    Returns
    -------
    Qd : ndarray
        Full discrete process covariance, shape (nx, nx).
    """
    if not Pinf.any():
        return Pinf
    return Pinf - Ad @ Pinf @ Ad.T


def diffusion_kron(A: np.ndarray, Q: np.ndarray, Ad: np.ndarray) -> np.ndarray:
    """Discretize diffusion matrix by solving indirectly the Lyapunov equation

    Charles C. Driver, Manuel C. Voelkle.
    Introduction to Hierarchical Continuous Time Dynamic Modelling With ctsem

    Parameters
    ----------
    A : array
        Continuous state matrix, shape (nx, nx).
    Q : array
        Full continuous diffusion covariance rate, shape (nx, nx), not a factor.
    Ad : array
        Discrete state matrix, shape (nx, nx).

    Returns
    -------
    Qd : array
        Full discrete process covariance, shape (nx, nx).
    """
    if not Q.any():
        return Q

    nx = Q.shape[0]
    Ix = np.eye(nx)
    A_sharp = np.kron(A, Ix) + np.kron(Ix, A)
    b = np.atleast_2d(Q.ravel()).T
    try:
        x = solve(-A_sharp, b)
    except (LinAlgWarning, LinAlgError):
        x = lstsq(-A_sharp, b, rcond=-1)[0]

    return diffusion_stationary(np.reshape(x, (nx, nx)), Ad)
