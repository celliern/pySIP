"""Registered post-fit stability, gradient, significance and correlation checks."""

from collections.abc import Callable
from typing import NamedTuple, TypeVar, cast

import numpy as np
import pandas as pd

from pysip.regressors import Regressor

Diagnostic = Callable[
    [Regressor, pd.DataFrame, np.ndarray],
    bool | np.bool_ | np.ma.core.MaskedConstant,
]
D = TypeVar("D", bound=Diagnostic)

s: list[Diagnostic] = []


def stack(f: D) -> D:
    """Register a diagnostic callable in insertion order and return it unchanged."""
    s.append(f)
    return f


@stack
def eigen_values(
    model: Regressor, summary: pd.DataFrame, corr: np.ndarray
) -> np.bool_:
    """Check that all continuous-time state-matrix eigenvalues have negative real parts."""
    return (-np.linalg.eigvals(model.ss.A).real > 0).all()


@stack
def summary(
    model: Regressor, summary: pd.DataFrame, corr: np.ndarray
) -> np.bool_:
    """Check that all absolute standardized-parameter gradients are below 1e-4."""
    return cast(np.bool_, (summary["|g(η)|"] < 1e-4).all())


@stack
def summary_penalty(
    model: Regressor, summary: pd.DataFrame, corr: np.ndarray
) -> np.bool_:
    """Check absolute physical-parameter penalty derivatives are below gradient magnitudes."""
    return cast(np.bool_, (summary["|dpen(θ)|"] < summary["|g(η)|"]).all())


@stack
def summary_pvalue(
    model: Regressor, summary: pd.DataFrame, corr: np.ndarray
) -> np.bool_:
    """Check that every parameter p-value is at most 0.05."""
    return cast(np.bool_, (summary["pvalue"] <= 5e-2).all())


@stack
def corr(
    model: Regressor, summary: pd.DataFrame, corr: np.ndarray
) -> np.bool_ | np.ma.core.MaskedConstant:
    """Check signed off-diagonal correlations are below 0.9, ignoring the diagonal."""
    non_diag = np.eye(corr.shape[0], dtype=bool)
    return (np.ma.masked_array(corr, mask=non_diag) < 0.9).all()


class Test(NamedTuple):
    """A diagnostic name, printable description and registered callable."""

    name: str
    description: str | None
    function: Diagnostic


tests: list[Test] = [Test(f.__name__, f.__doc__, f) for f in s]


def check_model(
    model: Regressor,
    summary: pd.DataFrame | None = None,
    corr: np.ndarray | None = None,
    verbose: bool = False,
    raise_error: bool = False,
) -> bool | np.bool_ | np.ma.core.MaskedConstant:
    """Run registered fit diagnostics and return their combined Boolean result.

    ``summary`` must contain ``|g(η)|``, ``|dpen(θ)|`` and ``pvalue`` columns,
    and ``corr`` must be the square parameter-correlation matrix. Their defaults
    do not compute missing results: diagnostics requiring them will fail.
    ``verbose`` prints colored diagnostic descriptions. ``raise_error`` raises
    RuntimeError on the first failed check; otherwise all checks are evaluated.
    """
    def _print(s: str) -> None:
        """Print a diagnostic message only in verbose mode."""
        if verbose:
            print(s)

    diag = True
    _print("")

    for idx, test in enumerate(tests):
        result = test.function(
            model, cast(pd.DataFrame, summary), cast(np.ndarray, corr)
        )
        diag = np.bitwise_and(diag, result)
        _print(
            ("\033[92m" if result else "\033[91m")
            + f"{idx}. {test.description}"
            + "\033[0m"
        )
        if raise_error and not result:
            raise RuntimeError(f"Failed {test.name}")

    return cast(bool | np.bool_ | np.ma.core.MaskedConstant, diag)
