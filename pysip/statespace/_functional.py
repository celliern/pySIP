"""Shared model equations evaluated with NumPy or optional immutable JAX arrays."""

from copy import copy
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
from scipy.linalg import expm

from ..utils.math import nearest_cholesky


class ContinuousMatrices(NamedTuple):
    """Continuous matrices with full covariances, rather than upper factors.

    A/B/C/D retain the model shapes; Qc is a diffusion rate, Rc is the
    measurement covariance, and P0c is the initial covariance. x0 is a column.
    Arrays can belong to NumPy or JAX without converting traced values to NumPy.
    """

    A: Any
    B: Any
    C: Any
    D: Any
    Qc: Any
    Rc: Any
    x0: Any
    P0c: Any


def array_namespace(array: Any) -> Any:
    """Return the numerical namespace without importing JAX for NumPy arrays."""
    if isinstance(array, np.ndarray):
        return np
    namespace = getattr(array, "__array_namespace__", None)
    if namespace is not None:
        return namespace()
    raise TypeError("Model matrices must be NumPy or JAX arrays")


def array_set(array: Any, index: Any, value: Any) -> Any:
    """Assign a matrix entry, preserving NumPy mutation and JAX immutability."""
    if isinstance(array, np.ndarray):
        array[index] = value
        return array
    xp = array_namespace(array)
    return array.at[index].set(xp.asarray(value, dtype=array.dtype))


def block_diag(*arrays: Any, xp: Any = np) -> Any:
    """Return a block diagonal matrix in the selected array namespace."""
    result = xp.zeros((sum(a.shape[0] for a in arrays), sum(a.shape[1] for a in arrays)))
    row = col = 0
    for array in arrays:
        rows, cols = array.shape
        result = array_set(result, (slice(row, row + rows), slice(col, col + cols)), array)
        row += rows
        col += cols
    return result


def model_covariances(model: Any) -> ContinuousMatrices:
    """Read current model matrices, converting upper factors to covariances."""
    return ContinuousMatrices(
        model.A, model.B, model.C, model.D, model.Q.T @ model.Q,
        model.R.T @ model.R, model.x0, model.P0.T @ model.P0,
    )


def apply_covariances(model: Any, matrices: ContinuousMatrices) -> None:
    """Write host matrices and upper covariance factors to a mutable model."""
    model.A, model.B, model.C, model.D = matrices[:4]
    model.Q = nearest_cholesky(np.asarray(matrices.Qc))
    model.R = nearest_cholesky(np.asarray(matrices.Rc))
    model.x0 = matrices.x0
    model.P0 = nearest_cholesky(np.asarray(matrices.P0c))


def evaluate_matrices(model: Any, theta: Any, xp: Any = np) -> ContinuousMatrices:
    """Evaluate a leaf's existing update equations on an independent matrix view.

    The view exposes physical parameters through ``parameters.theta``. Custom
    models using immutable array assignment can reuse this implementation;
    others can implement ``continuous_matrices`` directly. Original arrays and
    parameter objects are never modified, including when xp is NumPy.
    """
    view = copy(model)
    view.parameters = SimpleNamespace(theta=theta)
    for name in ("A", "B", "C", "D", "Q", "R", "x0", "P0"):
        setattr(view, name, xp.asarray(np.array(getattr(model, name), copy=True)))
    view.update_continuous_ssm()
    return model_covariances(view)


def matrix_exponential(array: Any, xp: Any = np) -> Any:
    """Evaluate a matrix exponential without importing the unused backend."""
    if xp is np:
        return expm(array)
    from jax.scipy.linalg import expm as jax_expm

    return jax_expm(array)


def discrete_covariance(
    matrices: ContinuousMatrices, dt: Any, hold_order: int, xp: Any = np,
) -> tuple[Any, Any, Any, Any]:
    """Integrate continuous input and diffusion covariances without PSD repair.

    Supports singular state matrices and zero process diffusion. Short-step
    Van Loan integration followed by doubling avoids unstable long-step
    exponentials. The JAX path uses a static loop so reverse AD is defined.
    """
    A, B, _, _, Qc, _, _, _ = matrices
    nx, nu = B.shape
    width = nx + (hold_order + 1) * nu
    augmented = xp.zeros((width, width), dtype=A.dtype)
    augmented = array_set(augmented, (slice(0, nx), slice(0, nx)), A)
    augmented = array_set(augmented, (slice(0, nx), slice(nx, nx + nu)), B)
    if hold_order:
        augmented = array_set(augmented, (slice(nx, nx + nu), slice(nx + nu, None)), xp.eye(nu))
    transition = matrix_exponential(augmented * dt, xp)
    Ad = transition[:nx, :nx]
    B0 = transition[:nx, nx:nx + nu]
    B1 = transition[:nx, nx + nu:] if hold_order else xp.zeros_like(B)

    van_loan = xp.concatenate((
        xp.concatenate((A, Qc), axis=1),
        xp.concatenate((xp.zeros_like(A), -A.T), axis=1),
    ), axis=0)
    magnitude = xp.linalg.norm(A, ord=np.inf) * xp.abs(dt)
    squarings = xp.maximum(0, xp.ceil(xp.log2(xp.maximum(magnitude, 1)))).astype(int)
    short = matrix_exponential(van_loan * (dt / 2.0**squarings), xp)
    a = short[:nx, :nx]
    q = short[:nx, nx:] @ a.T
    if xp is np:
        for _ in range(int(squarings)):
            q = q + a @ q @ a.T
            a = a @ a
    else:
        from jax import lax

        def double(i: Any, carry: tuple[Any, Any]) -> tuple[Any, Any]:
            def step(values: tuple[Any, Any]) -> tuple[Any, Any]:
                a, q = values
                return a @ a, q + a @ q @ a.T
            return lax.cond(i < squarings, step, lambda values: values, carry)

        # ponytail: a bounded differentiable loop; larger scales fail explicitly.
        a, q = lax.fori_loop(0, 64, double, (a, q))
        q = xp.where(squarings <= 64, q, xp.full_like(q, xp.nan))
    return Ad, B0, B1, (q + q.T) / 2.0
