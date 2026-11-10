"""Matplotlib visualizations for predictive samples and residual diagnostics."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

Color = str | tuple[float, float, float] | tuple[float, float, float, float]


def percentile_plot(
    x: ArrayLike,
    y: np.ndarray,
    n: int = 4,
    percentile_min: float = 2.5,
    percentile_max: float = 97.5,
    color: Color = "darkred",
    plot_mean: bool = False,
    plot_median: bool = True,
    line_color: Color = "maroon",
    **kwargs: Any,
) -> Axes:
    """Draw predictive percentile bands on and return the current axes.

    ``x`` has shape (t,) and ``y`` has shape (samples, t). ``n`` pairs of
    percentile curves span ``percentile_min`` to ``percentile_max`` around
    50; percentile values must lie in [0, 100]. ``plot_mean`` and
    ``plot_median`` add central curves in ``line_color``. ``color`` fills
    bands; ``kwargs['alpha']`` overrides opacity (default 2/n); remaining
    keyword arguments are ignored. The current axes are mutated, not the
    inputs. NumPy and Matplotlib validate percentile and plotting arguments.
    """

    # calculate the lower and upper percentile groups, skipping 50 percentile
    linspace1 = np.linspace(percentile_min, 50, num=n, endpoint=False)
    linspace2 = np.linspace(50, percentile_max, num=n + 1)[1:]
    perc1 = np.percentile(y, linspace1, axis=0)
    perc2 = np.percentile(y, linspace2, axis=0)

    # transparency
    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    else:
        alpha = 1 / (n / 2)

    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)

    if plot_mean:
        plt.plot(x, np.mean(y, axis=0), color=line_color)

    if plot_median:
        plt.plot(x, np.median(y, axis=0), color=line_color)

    return plt.gca()


def plot_ccf(
    lags: np.ndarray,
    coeffs: np.ndarray,
    confidence: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Draw CCF stems and confidence bands, returning the modified axes.

    ``lags``, ``coeffs`` and positive half-width ``confidence`` are matching
    shape-(n,) arrays returned by `pysip.utils.statistics.ccf`.
    Omitted ``ax`` creates a figure and axes. Input arrays are not modified.
    """
    if ax is None:
        _, ax = plt.subplots()

    _, stemlines, baseline = ax.stem(lags, coeffs, linefmt="-", markerfmt="None")
    ax.fill_between(
        lags, -confidence, confidence, facecolor="darkred", edgecolor="None", alpha=0.3
    )

    plt.setp(baseline, color="darkblue")
    plt.setp(stemlines, color="darkblue")

    return ax


def plot_cpgram(
    y: np.ndarray, freq: np.ndarray, crit: float, ax: Axes | None = None
) -> Axes:
    """Draw a cumulative periodogram, returning the modified axes.

    ``y`` and ``freq`` are matching shape-(n,) power and frequency arrays;
    ``crit`` is the scalar confidence half-width from
    `pysip.utils.statistics.cpgram`. Omitted ``ax`` creates a figure and
    axes. Limits are set to [0, 0.5] by [0, 1]. Inputs are not modified;
    zero total power produces nonfinite cumulative values.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(freq, np.cumsum(y) / np.sum(y), color="darkblue")
    ax.fill_between(
        freq,
        2 * freq - crit,
        2 * freq + crit,
        facecolor="darkred",
        edgecolor="None",
        alpha=0.3,
    )
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    return ax
