import pickle
from copy import deepcopy
from typing import ClassVar

import numpy as np
import pytest
from scipy import stats

from pysip.params.prior import (
    Beta,
    Gamma,
    InverseGamma,
    LogNormal,
    Normal,
    PriorMeta,
    Uniform,
)


@pytest.fixture(name="N")
def n_samples():
    return 10000


@pytest.fixture
def random_seed():
    state = np.random.get_state()
    np.random.seed(0)
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("Prior", [Normal, Gamma, Beta, InverseGamma, LogNormal])
def test_prior_equality(Prior):
    assert Prior() == Prior()


@pytest.mark.parametrize(
    "Prior, log, scipy_fun",
    [
        (Normal, -0.9639385332046727, stats.norm(loc=0, scale=1)),
        (Beta, 0.27990188513281833, stats.beta(a=3, b=3)),
        (Gamma, -3.4010927892118175, stats.gamma(a=3, scale=1 / 1)),
        (InverseGamma, 0.7894107034104656, stats.invgamma(a=3, scale=1)),
        (LogNormal, -0.43974098565696607, stats.lognorm(loc=0, s=1)),
    ],
)
@pytest.mark.usefixtures("random_seed")
def test_log_pdf(Prior, log, scipy_fun, N):
    prior = Prior()
    rvs = prior.random(N)
    x = np.random.uniform()

    assert prior.logpdf(0.3) == pytest.approx(log)
    assert prior.mean == pytest.approx(np.mean(rvs), abs=3 * np.std(rvs) / np.sqrt(N))
    assert prior.logpdf(x) == pytest.approx(scipy_fun.logpdf(x))


@pytest.mark.parametrize(
    "prior, updates",
    [
        (Normal(), {"mu": 0.5, "sigma": 2.0}),
        (Gamma(), {"alpha": 2.5, "beta": 1.5}),
        (Beta(), {"alpha": 2.5, "beta": 1.5}),
        (InverseGamma(), {"alpha": 2.5, "beta": 1.5}),
        (LogNormal(), {"mu": 0.5, "sigma": 2.0}),
        (Uniform(), {"lower": 0.2, "upper": 2.0}),
    ],
)
def test_logpdf_tracks_mutation_and_copies(prior, updates):
    x = np.array([-0.1, 0.3, 0.8, 2.5])
    original = prior.scipy_dist.logpdf(x)
    np.testing.assert_array_equal(prior.logpdf(x), original)

    for name, value in updates.items():
        setattr(prior, name, value)
    expected = prior.scipy_dist.logpdf(x)
    assert not np.array_equal(expected, original)
    for copied in (prior, deepcopy(prior), pickle.loads(pickle.dumps(prior))):
        np.testing.assert_array_equal(copied.logpdf(x), expected)
        assert copied.logpdf(0.3) == copied.scipy_dist.logpdf(0.3)


def test_custom_factory_observes_external_mutation():
    offset = 0.0

    def factory(mu, sigma):
        return stats.norm(loc=mu + offset, scale=sigma)

    class CustomNormal(Normal, metaclass=PriorMeta):
        mu: float = 0.0
        sigma: float = 1.0
        _scipy_factory: ClassVar = factory
        _pymc_factory: ClassVar = Normal._pymc_factory

    prior = CustomNormal()
    assert prior.logpdf(0.3) == stats.norm.logpdf(0.3)
    offset = 2.0
    assert prior.logpdf(0.3) == stats.norm.logpdf(0.3, loc=2.0)
    prior.mu = -0.5
    assert prior.logpdf(0.3) == stats.norm.logpdf(0.3, loc=1.5)
