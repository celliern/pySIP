import random

import numpy as np
import pytest
from scipy.stats import t

from pysip.utils.statistics import (
    ccf,
    check_ccf,
    check_cpgram,
    cpgram,
    ttest,
)


def test_ccf():
    lags, correlation_coeffs, confidence = ccf([1, 2, 3], [4, 5, 6])
    np.testing.assert_array_equal(lags, np.array([0, 1, 2]))
    np.testing.assert_array_equal(correlation_coeffs, np.array([1.0, 0.0, -0.5]))
    np.testing.assert_allclose(
        confidence, np.array([1.13158573, 1.13158573, 1.13158573])
    )


@pytest.mark.parametrize("n_lags", [0, 1, 2])
def test_ccf_truncated_lags(n_lags):
    lags, coefficients, confidence = ccf([1, 2, 3, 4], n_lags=n_lags)
    np.testing.assert_array_equal(lags, np.arange(n_lags + 1))
    assert coefficients.shape == confidence.shape == lags.shape
    assert isinstance(check_ccf(lags, coefficients, confidence)[0], (bool, np.bool_))


def test_ttest_two_sided_negative_estimates():
    estimates = np.array([-2.0, 2.0, -20.0, 20.0])
    actual = ttest(estimates, np.ones(4), 104)
    np.testing.assert_allclose(actual, 2 * t.sf(np.abs(estimates), 100))
    assert np.all((actual >= 0) & (actual <= 1))
    assert actual[0] == actual[1]
    assert actual[2] == actual[3] > 0


def test_ccf_raises():
    with pytest.raises(ValueError):
        ccf([1, 2, 3], [])

    with pytest.raises(ValueError):
        ccf([1, 2, 3], [4, 5, 6], n_lags=-2)

    with pytest.raises(ValueError):
        ccf([1, 2, 3], [4, 5, 6], n_lags=20)


def test_check_ccf():
    lags, correlation_coeffs, confidence = ccf([1, 2, 3], [4, 5, 6])
    assert check_ccf(lags, correlation_coeffs, confidence)[0]


def test_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    assert isinstance(y, np.ndarray) and len(y) == 1399
    assert isinstance(freq, np.ndarray) and len(freq) == 1399
    assert crit == pytest.approx(0.0361880988)


def test_check_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    assert not check_cpgram(y, freq, crit)[0]
