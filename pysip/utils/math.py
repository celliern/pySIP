"""Stable scalar transforms, error metrics and covariance-factor utilities."""

import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import LinAlgError, LinAlgWarning, ldl, solve_triangular


def log1p_exp(a: float) -> float:
    """Return scalar ``log(1 + exp(a))`` without exponential overflow."""
    if a > 0.0:
        return a + np.log1p(np.exp(-a))
    return np.log1p(np.exp(a))


def log_sum_exp(a: float, b: float) -> float:
    """Return scalar ``log(exp(a) + exp(b))`` stably, including infinite inputs."""
    if np.isneginf(a):
        return b
    if np.isposinf(a) and np.isposinf(b):
        return np.inf
    if a > b:
        return a + log1p_exp(b - a)
    return b + log1p_exp(a - b)


def cholesky_inverse(matrix: np.ndarray) -> np.ndarray:
    """Return a lower factor ``L`` with ``L @ L.T = inv(matrix)``.

    ``matrix`` is a positive-definite covariance of shape ``(..., n, n)``;
    leading batch dimensions require support from SciPy's triangular solver.
    The input is not modified. Non-positive-definite matrices raise LinAlgError.
    Adapted from numpyro's distribution utilities.
    """
    tril_inv = np.swapaxes(
        np.linalg.cholesky(matrix[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = np.broadcast_to(np.identity(matrix.shape[-1]), tril_inv.shape)
    return solve_triangular(tril_inv, identity, lower=True)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return the difference norm divided by sqrt(len(y)) for matching arrays."""
    return np.linalg.norm(y - yhat) / np.sqrt(len(y))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return mean absolute error over all elements of matching arrays."""
    return np.mean(np.abs(y - yhat))


def mad(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return maximum absolute error over matching nonempty arrays."""
    return np.max(np.abs(y - yhat))


def ned(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return the difference norm divided by summed norms; zero norms yield NaN."""
    return np.linalg.norm(y - yhat) / (np.linalg.norm(y) + np.linalg.norm(yhat))


def smape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return mean symmetric absolute error in percent for matching arrays.

    Each term divides by ``abs(y) + abs(yhat)``; paired zeros produce NaN.
    No input is modified.
    """
    return 100.0 * np.mean(np.abs(y - yhat) / np.add(np.abs(y), np.abs(yhat)))


def fit(y: np.ndarray, yhat: np.ndarray) -> float:
    """Return ``1 - ||y-yhat||₂ / ||y-mean(y)||₂`` for matching 1-D series.

    This is a fraction, not a percentage. Constant reference series cause a
    zero denominator; no special handling or mutation is performed.
    """
    return 1.0 - np.linalg.norm(y - yhat, 2) / np.linalg.norm(y - np.mean(y), 2)


def nearest_cholesky(m: np.ndarray, method: str = "ldl") -> np.ndarray | None:
    """Approximate a covariance factor from a real square matrix ``m``.

    The input is symmetrized without mutation. ``eigen`` clips negative
    eigenvalues and returns an upper factor ``R`` with covariance ``R.T @ R``.
    ``ldl`` first tries Cholesky, then clips negative eigenvalues of the
    block-diagonal LDL factor for semidefinite or indefinite inputs.
    Pivoted and 2-by-2 block factors are triangularized by QR, preserving
    the covariance in the original state ordering.
    ``jitter`` tries an upper Cholesky factor with diagonal additions from
    1e-9 up to (but excluding) 1, returning None if all attempts fail.
    An all-zero input is returned unchanged, including its object identity.
    Invalid methods raise ValueError except on that all-zero fast path.
    """
    if not m.any():
        return m

    x = (m + m.T) / 2.0
    if method == "ldl":
        try:
            return np.linalg.cholesky(x).T
        except LinAlgError:
            pass
        lu, d, perm = ldl(x, lower=True, hermitian=False)
        if np.count_nonzero(d) == np.count_nonzero(d.diagonal()):
            factor = np.sqrt(np.maximum(d.diagonal(), 0.0))[:, None] * lu.T
            if np.array_equal(perm, np.arange(x.shape[0])):
                return factor
        else:
            eigvals, vectors = np.linalg.eigh(d)
            factor = np.sqrt(np.maximum(eigvals, 0.0))[:, None] * vectors.T @ lu.T
        return np.linalg.qr(factor, mode="r")
    elif method == "eigen":
        eigvals, v = np.linalg.eigh(x)
        return np.linalg.qr(
            np.diag([np.sqrt(w) if w > 0 else 0 for w in eigvals]) @ v.T, "r"
        )
    elif method == "jitter":
        Ix = np.eye(x.shape[0])
        jitter = 1e-9
        while jitter < 1.0:
            try:
                chol = np.linalg.cholesky(x + jitter * Ix).T
                return chol
            except (LinAlgError, LinAlgWarning):
                jitter *= 10.0
    else:
        raise ValueError("`method` must be set to `ldl`, `eigen` or `jitter`")


def diff_upper_cholesky(R: np.ndarray, dS: np.ndarray) -> np.ndarray:
    """Differentiate an upper Cholesky factor without modifying the inputs.

    $$
    S = R^T R
    $$

    Parameters
    ----------
    R : numpy.ndarray
        Upper triangular factor of S, shape (n, n).
    dS : numpy.ndarray
        Symmetric covariance derivative, shape (n, n).

    Returns
    -------
    numpy.ndarray
        Upper-factor derivative, shape (n, n). Singular solves use least squares;
        other solve failures propagate without attempting the next solve.

    References
    ----------
    Iain Murray, *Differentiation of the Cholesky decomposition*, February 2016.
    https://homepages.inf.ed.ac.uk/imurray2/pub/16choldiff/choldiff.pdf
    """

    def Phi(x: np.ndarray) -> np.ndarray:
        """Copy the upper triangle and halve its diagonal."""
        x = np.triu(x)
        x[np.diag_indices_from(x)] *= 0.5
        return x

    try:
        right_ = solve_triangular(R, dS.T, trans=1).T
    except (LinAlgError, LinAlgWarning):
        right_ = np.transpose(lstsq(R.T, dS.T, rcond=-1)[0])
    try:
        left_ = solve_triangular(R, right_, trans=1)
    except (LinAlgError, LinAlgWarning):
        left_ = lstsq(R.T, right_, rcond=-1)[0]
    return Phi(left_) @ R


def time_series_pca(X: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Project ``X`` of shape ``(n, nc)`` onto one covariance-PCA component.

    Eigenvectors are computed from column-standardized data, but the returned
    shape-(n,) series projects the original data using weights normalized to
    sum to one. The input is not modified. Constant columns or a zero sum of
    weights may produce nonfinite results or a linear-algebra error.
    ``verbose`` prints the first eigenvalue's fraction of total variance
    (the printed percent sign does not multiply this fraction by 100).
    """
    w, v = np.linalg.eigh(np.cov((X - X.mean(axis=0)) / X.std(axis=0), rowvar=False))
    idx = w.argsort()[::-1]
    w, v = w[idx], v[:, idx]
    if verbose:
        print(f"explain {np.squeeze(w[0] / w.sum()):.3e} % of the variance")
    return X @ (v[:, 0] / v[:, 0].sum())
