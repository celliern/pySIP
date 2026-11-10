"""Synthetic excitation signals and regularly or randomly sampled sine waves."""

import itertools

import numpy as np
from numpy.typing import NDArray
from pylfsr import LFSR


def prbs(Tmax: int, Tmin: int, initstate: str = "random") -> np.ndarray:
    """Return a one-dimensional pseudo-random binary excitation sequence.

    Each register bit is repeated ``Tmin`` times. With register length
    ``r = ceil(Tmax/Tmin)``, the result has ``(2**r - 1) * Tmin`` entries.
    ``initstate`` is ``'random'`` or ``'ones'``. Noninteger durations raise
    TypeError; Tmax < 2, Tmin < 1, Tmin >= Tmax, invalid initial states or
    register lengths outside [2, 31] raise ValueError. Assertions check
    that maximum and minimum generated runs equal Tmax and Tmin, so otherwise
    admissible durations or random initial states may still raise AssertionError.
    The external LFSR implementation determines the binary array's dtype.
    """
    if not isinstance(Tmax, int):
        raise TypeError("`Tmax` must be an integer")

    if Tmax < 2:
        raise ValueError("`Tmax` must be > 2")

    if not isinstance(Tmin, int):
        raise TypeError("`Tmax` must be an integer")

    if Tmin < 1:
        raise ValueError("`Tmin` must be > 1")

    if Tmin >= Tmax:
        raise ValueError("`Tmax` must be strictly superior to `Tmin`")

    __init_availabble__ = ["random", "ones"]
    if initstate not in __init_availabble__:
        raise ValueError(f"`initstate` must be either {__init_availabble__}")

    # get the register length
    n = np.ceil(Tmax / Tmin)
    if n < 2 or n > 31:
        raise ValueError(
            "The PRBS cannot be generated, " "decompose the signal in two sequences"
        )

    # Linear feedback register up to 32 bits
    fpoly = {
        2: [2, 1],
        3: [3, 1],
        4: [4, 1],
        5: [5, 2],
        6: [6, 1],
        7: [7, 1],
        8: [8, 4, 3, 2],
        9: [9, 4],
        10: [10, 3],
        11: [11, 2],
        12: [12, 6, 4, 1],
        13: [13, 4, 3, 1],
        14: [14, 8, 6, 1],
        15: [15, 1],
        16: [16, 12, 3, 1],
        17: [17, 3],
        18: [18, 7],
        19: [19, 5, 2, 1],
        20: [20, 3],
        21: [21, 2],
        22: [22, 1],
        23: [23, 5],
        24: [24, 7, 2, 1],
        25: [25, 3],
        26: [26, 6, 2, 1],
        27: [27, 5, 2, 1],
        28: [28, 3],
        29: [29, 2],
        30: [30, 23, 2, 1],
        31: [31, 3],
    }

    L = LFSR(fpoly=fpoly[n], initstate=initstate, verbose=False)

    seq = []
    for n in range(L.expectedPeriod):
        L.next()
        seq.append(L.state[0])

    seq_padded = np.repeat(seq, Tmin)

    # check generated PRBS
    assert seq_padded.shape[0] == L.expectedPeriod * Tmin
    assert max(len(list(v)) for g, v in itertools.groupby(seq_padded)) == Tmax
    assert min(len(list(v)) for g, v in itertools.groupby(seq_padded)) == Tmin

    return seq_padded


def generate_time(
    n: int = 50, bounds: tuple[float, float] = (0, 50), random: bool = False
) -> NDArray[np.float64]:
    """Return a sorted shape-(n,) float time array between increasing bounds.

    Linear spacing includes both endpoints; random spacing uses sorted uniform
    draws from the half-open interval [lower, upper). Wrong argument types
    or nonincreasing bounds raise ValueError. Negative n is rejected by NumPy.
    A tuple with fewer than two bounds raises IndexError.
    """
    if not isinstance(n, int):
        raise ValueError("`n` must be an integer")  # noqa: TRY004 - Documented ValueError contract.

    if not isinstance(bounds, tuple):
        raise ValueError("`bounds` must be a tuple")  # noqa: TRY004 - Documented ValueError contract.

    bounds_array = np.asarray(bounds)
    if bounds_array[1] <= bounds_array[0]:
        raise ValueError("upper bound > lower bound")

    if not isinstance(random, bool):
        raise ValueError("`random` must be a boolean")  # noqa: TRY004 - Documented ValueError contract.

    if random:
        t = np.sort(
            bounds_array[0]
            + (bounds_array[1] - bounds_array[0]) * np.random.rand(n)
        )
    else:
        t = np.linspace(bounds_array[0], bounds_array[1], num=n)

    return t


def generate_sine(
    n: int = 50,
    bounds: tuple[float, float] = (0, 50),
    random: bool = False,
    period: float = 24.0,
    phase: float = 0.0,
    amplitude: float = 10.0,
    offset: float = 0.0,
    noise_std: float = 1.0,
    clip_to_0: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(time, values)`` arrays of shape (n,) for a noisy sine wave.

    Time sampling follows `generate_time`. Values equal
    ``offset + amplitude*sin(2*pi*time/period + phase)`` plus independent
    Gaussian noise with standard deviation ``noise_std``. ``phase`` is in
    radians with no upper bound. ``clip_to_0`` clips negative generated values
    in place before returning them; no caller input is modified.

    ``period`` and ``amplitude`` must be positive floats; ``phase`` and
    ``noise_std`` must be nonnegative floats; ``offset`` must be a float.
    Violations raise ValueError. Time-argument validation is delegated to
    `generate_time`.
    """
    if not isinstance(period, float) or period <= 0.0:
        raise ValueError("`period` must be a positive float")

    if not isinstance(amplitude, float) or amplitude <= 0.0:
        raise ValueError("`amplitude` must be a positive float")

    if not isinstance(noise_std, float) or noise_std < 0.0:
        raise ValueError("`noise_std` must be a positive float")

    if not isinstance(phase, float) or phase < 0.0:
        raise ValueError("`phase` must be a positive float")

    if not isinstance(offset, float):
        raise ValueError("`offset` must be a float")  # noqa: TRY004 - Documented ValueError contract.

    t = generate_time(n, bounds, random)
    y = (
        offset
        + amplitude * np.sin(2.0 * np.pi / period * t + phase)
        + noise_std * np.random.randn(t.shape[0])
    )

    if clip_to_0:
        y[y < 0] = 0

    return t, y


def generate_random_binary(
    n: int = 50, bounds: tuple[float, float] = (0, 1e3)
) -> np.ndarray:
    """Return shape-(max(n, 0),) independent draws from the two bounds.

    Each endpoint is selected with equal probability. The output dtype follows
    the bounds' numeric dtype, except empty output is floating point. Noninteger
    n, nontuple bounds or nonincreasing bounds raise ValueError. A tuple with
    fewer than two entries raises IndexError; negative n returns an empty array.
    """
    if not isinstance(n, int):
        raise ValueError("`n` must be an integer")  # noqa: TRY004 - Documented ValueError contract.

    if not isinstance(bounds, tuple):
        raise ValueError("`bounds` must be a tuple")  # noqa: TRY004 - Documented ValueError contract.

    bounds_array = np.asarray(bounds)
    if bounds_array[1] <= bounds_array[0]:
        raise ValueError("upper bound > lower bound")

    return np.asarray(
        [
            bounds_array[1] if np.random.randn() > 0 else bounds_array[0]
            for _ in range(n)
        ]
    )
