"""Likelihood criteria and residual correlation and spectral diagnostics."""
from collections.abc import Sequence

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray
from scipy.signal import correlate


def lrtest(
    loglik: float, loglik_sub: float, n_par: int, n_par_sub: int, negative: bool = True
) -> float:
    """Return a chi-square likelihood-ratio p-value for nested models.

    ``loglik`` and ``loglik_sub`` are float log-likelihoods for the full and
    restricted model, negated when ``negative`` is True. ``n_par`` and
    ``n_par_sub`` are integer free-parameter counts; their difference is the
    chi-square degrees of freedom. Equal counts are accepted (yielding NaN).
    Incorrect scalar types raise TypeError, and ``n_par_sub > n_par`` raises
    ValueError.
    """
    if not isinstance(loglik, float):
        raise TypeError("`loglik` must be a float")

    if not isinstance(loglik_sub, float):
        raise TypeError("`loglik_sub` must be a float")

    if not isinstance(n_par, int):
        raise TypeError("`n_par` must be an integer")

    if not isinstance(n_par_sub, int):
        raise TypeError("`n_par_sub` must be an integer")

    if not isinstance(negative, bool):
        raise TypeError("`negative` must be a boolean")

    if n_par_sub > n_par:
        raise ValueError(
            "The sub-model must have less parameters " "than the larger model"
        )

    if negative:
        lrt = loglik - loglik_sub
    else:
        lrt = loglik_sub - loglik

    return sp.stats.chi2.sf(-2.0 * lrt, n_par - n_par_sub)


def aic(loglik: float, n_par: int, negative: bool = True) -> float:
    """Return scalar Akaike information criterion ``2*k - 2*loglik``.

    ``n_par`` is integer k; ``negative`` indicates a negated float likelihood.
    Incorrect scalar types raise TypeError. Counts are not range-validated.
    """
    if not isinstance(loglik, float):
        raise TypeError("`loglik` must be a float")

    if not isinstance(n_par, int):
        raise TypeError("`n_par` must be an integer")

    if not isinstance(negative, bool):
        raise TypeError("`negative` must be a boolean")

    if negative:
        return 2.0 * n_par + 2.0 * loglik
    return 2.0 * n_par - 2.0 * loglik


def ttest(theta: ArrayLike, sigma: ArrayLike, N: int) -> NDArray[np.float64]:
    """Return two-sided t-test p-values for zero physical parameter values.

    ``theta`` and corresponding standard errors ``sigma`` are shape-(p,)
    array-like inputs. The returned shape-(p,) float array uses ``N - p``
    degrees of freedom, where N is the observation count. Standard errors
    at or below zero leave NaN entries; invalid degrees of freedom also yield
    NaN. Inputs are not modified and no significance threshold is applied.
    """
    if not isinstance(theta, np.ndarray):
        theta = np.array(theta)

    if not isinstance(sigma, np.ndarray):
        sigma = np.array(sigma)

    idx = sigma > 0
    if np.any(idx is False):
        print("Negative standard deviation in the t-test, the p-value is NaN")

    Np = len(theta)
    pvalue = np.full(Np, np.nan)
    pvalue[idx] = 2 * sp.stats.t.sf(np.abs(theta[idx] / sigma[idx]), N - Np)

    return pvalue


def ccf(
    x: np.ndarray,
    y: np.ndarray | None = None,
    n_lags: int | None = None,
    ci: float = 0.95,
) -> tuple[NDArray[np.int_], np.ndarray, NDArray[np.float64]]:
    """Return nonnegative-lag normalized correlations and normal confidence bounds.

    ``x`` and ``y`` are equal-length shape-(N,) series; omitted y uses x
    (autocorrelation). ``n_lags`` defaults to N-1 and must satisfy
    ``0 <= n_lags < N``. Invalid lengths or lag counts raise ValueError.
    ``ci`` is the normal interval probability, conventionally between 0 and 1
    (not validated). The tuple contains lags, coefficients and positive
    confidence half-widths, each shape ``(n_lags + 1,)``. Constant series
    divide by zero. No inputs are modified.
    """

    if y is None:
        y = x

    N = len(x)
    n_lags = N - 1 if n_lags is None else int(n_lags)

    if len(x) != len(y):
        raise ValueError("x and y must have equal length")

    if n_lags >= N or n_lags < 0:
        raise ValueError(f"maxlags must belong to [1 {n_lags - 1}]")

    lags = np.arange(n_lags + 1)

    correlation_coeffs = correlate(x - np.mean(x), y - np.mean(y)) / (
        np.std(x) * np.std(y) * N
    )

    cut = range(N - 1, N + n_lags)

    # confidence interval lines
    confidence = np.ones(len(cut)) * sp.stats.norm.ppf((1 + ci) / 2) / np.sqrt(N)

    return lags, correlation_coeffs[cut], confidence


def check_ccf(
    lags: np.ndarray,
    coeffs: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.95,
) -> tuple[np.bool_, float]:
    """Return ``(passed, fraction_in_band)`` for matching shape-(n,) CCF arrays.

    Coefficients must satisfy ``abs(coeffs) < confidence`` to count in-band;
    passing requires a fraction at least ``threshold`` (normally in [0, 1],
    not validated). ``lags`` is retained for the CCF calling convention but
    is not used. Inputs are not modified; empty coefficients divide by zero.
    """
    in_band = np.sum(np.abs(coeffs) < confidence) / len(coeffs)
    return in_band >= threshold, in_band


def cpgram(
    ts: Sequence[float] | np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return ``(power, frequencies, critical_width)`` for a residual series.

    For a shape-(N,) real series, power is squared full-FFT magnitude divided
    by N. An even N drops the final FFT bin, so both arrays have length N-1
    for even N and N otherwise. Frequencies are linearly spaced from 0 to
    0.5; they are not the FFT's native frequency bins. The scalar width uses
    the 95% cumulative-periodogram approximation. No cumulative sum is
    applied here and the input is not modified.
    """
    spectrum = np.fft.fft(ts)
    n = len(ts)
    y = (np.sqrt(spectrum.real**2 + spectrum.imag**2)) ** 2 / n
    if n % 2 == 0:
        n -= 1
        y = y[:n]

    freq = np.linspace(0, 0.5, n, endpoint=True)
    crit = 1.358 / (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))

    return y, freq, crit


def check_cpgram(
    y: np.ndarray,
    freq: np.ndarray,
    crit: float,
    threshold: float = 0.95,
) -> tuple[np.bool_, float]:
    """Return ``(passed, fraction_in_band)`` for periodogram arrays of shape (n,).

    ``y``, ``freq`` and scalar ``crit`` are the outputs of `cpgram`.
    A point is in-band when ``abs(cumsum(y)/sum(y) - 2*freq) < crit``;
    passing requires at least ``threshold`` (normally [0, 1], not validated).
    No inputs are modified. Zero total power or empty input divides by zero.
    """
    cum_sum = np.cumsum(y) / np.sum(y) - 2 * freq
    in_band = np.sum(np.abs(cum_sum) < crit) / len(freq)
    return in_band >= threshold, in_band
