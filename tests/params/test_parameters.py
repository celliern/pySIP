import numpy as np
import pytest

from pysip.params import Parameters


@pytest.fixture
def parameters():
    return Parameters(["a", "b", "c"], name="alpha")


@pytest.fixture
def parameters_beta():
    return Parameters(["c", "d", "e"], name="beta")


@pytest.fixture
def parameters_with_transforms():
    parameters = [
        {"name": "a", "value": 1.0, "transform": "log"},
        {"name": "b", "value": 2.0, "transform": "logit", "bounds": (1.0, 3.0)},
    ]

    return Parameters(parameters=parameters, name="transform")


def test_parameters(parameters):
    assert np.allclose(parameters.theta, [0, 0, 0])


def test_repr(parameters, parameters_beta, parameters_with_transforms):
    print(parameters.__repr__())
    assert (
        parameters.__repr__()
        == """Parameters alpha
Parameter(name='a', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
Parameter(name='b', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
Parameter(name='c', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
"""
    )

    s = parameters + parameters_beta + parameters_with_transforms
    assert (
        s.__repr__()
        == """Parameters alpha__beta__transform
* alpha__beta
    * alpha
        Parameter(name='a', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
        Parameter(name='b', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
        Parameter(name='c', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
    * beta
        Parameter(name='c', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
        Parameter(name='d', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
        Parameter(name='e', value=0.0, loc=0.0, scale=1.0, bounds=(None, None), transform=none, prior=None)
* transform
    Parameter(name='a', value=1.0, loc=0.0, scale=1.0, bounds=(None, None), transform=log, prior=None)
    Parameter(name='b', value=2.0, loc=0.0, scale=1.0, bounds=(1.0, 3.0), transform=logit, prior=None)
"""
    )


def test_transform(parameters_with_transforms):
    assert parameters_with_transforms.eta == pytest.approx([0.0, 0.0])


def test_set_parameter(parameters):
    assert parameters.parameters[0].name == "a"
    assert parameters.parameters[0].value == 0.0

    parameters.set_parameter("a", value=3.0)

    assert parameters.parameters[0].value == 3.0


def test_set_theta(parameters):
    assert np.allclose(parameters.theta, [0, 0, 0])
    parameters.theta = [1.0, 0.0, 0.0]
    assert np.allclose(parameters.theta, [1.0, 0.0, 0.0])


def test_add(parameters, parameters_with_transforms):
    new_parameters = parameters + parameters_with_transforms

    assert isinstance(new_parameters, Parameters)
    assert np.allclose(new_parameters.theta, [0.0, 0.0, 0.0, 1.0, 2.0])


def test_add_recursive(parameters, parameters_beta, parameters_with_transforms):
    new_parameters = parameters + parameters_beta
    new_parameters = new_parameters + parameters_with_transforms

    assert isinstance(new_parameters, Parameters)
    assert np.allclose(new_parameters.theta, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])


def test_add_set_theta(parameters, parameters_beta):
    new_parameters = parameters + parameters_beta

    assert np.allclose(new_parameters.theta, [0, 0, 0, 0, 0, 0])

    new_parameters.theta = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0]

    assert np.allclose(new_parameters.theta, [1.0, 0.0, 0.0, 2.0, 3.0, 0.0])
    assert np.allclose(parameters.theta, [1.0, 0.0, 0.0])
    assert np.allclose(parameters_beta.theta, [2.0, 3.0, 0.0])


def test_add_set_parameters(parameters, parameters_with_transforms):
    parameters = parameters + parameters_with_transforms

    assert parameters.parameters[0].name == "a"
    assert parameters.parameters[0].value == 0.0

    parameters.set_parameter("alpha", "a", value=3.0)

    assert parameters.parameters[0].value == 3.0


@pytest.fixture
def mixed_parameters():
    return Parameters([
        {"name": "fixed", "value": 3.0, "transform": "fixed"},
        {"name": "free", "value": 2.0, "loc": 10.0, "scale": 4.0, "transform": "log"},
    ], name="mixed")


def test_all_and_free_vector_setters(mixed_parameters):
    parameters = mixed_parameters
    parameters.eta = [7.0, np.log(3.0)]
    np.testing.assert_allclose(parameters.eta, [7.0, np.log(3.0)])
    np.testing.assert_allclose(parameters.theta, [7.0, 22.0])
    parameters.theta_sd = [5.0]
    np.testing.assert_allclose(parameters.theta_sd, [5.0])
    np.testing.assert_allclose(parameters.eta, [7.0, np.log(5.0)])
    parameters.theta_free = [34.0]
    np.testing.assert_allclose(parameters.theta, [7.0, 34.0])
    parameters.eta_free = [np.log(2.0)]
    np.testing.assert_allclose(parameters.theta, [7.0, 18.0])
    parameters.theta = [9.0, 26.0]
    np.testing.assert_allclose(parameters.eta, [9.0, np.log(4.0)])
    assert parameters.ids_free == ["free"]


@pytest.mark.parametrize("attribute", ["theta", "eta", "theta_free", "eta_free", "theta_sd"])
def test_vector_setters_reject_wrong_lengths(mixed_parameters, attribute):
    expected = 2 if attribute in ("theta", "eta") else 1
    original = mixed_parameters.theta.copy()
    for size in (expected - 1, expected + 1):
        with pytest.raises(ValueError, match=f"{expected} are expected"):
            setattr(mixed_parameters, attribute, [1.0] * size)
        np.testing.assert_array_equal(mixed_parameters.theta, original)


@pytest.mark.parametrize("n_free", [0, 1, 2])
@pytest.mark.parametrize("n_init", [1, 3])
@pytest.mark.parametrize("method", ["unconstrained", "zero", "fixed", "prior", "value"])
def test_initialization_preserves_parameter_axis(n_free, n_init, method):
    parameters = Parameters([
        {"name": "fixed", "value": 7.0, "transform": "fixed"},
        *[{"name": f"p{i}", "value": 1.0} for i in range(n_free)],
    ])
    initial = parameters.init_parameters(n_init, method)
    assert initial.shape == ((n_free,) if n_init == 1 else (n_free, n_init))
    assert parameters.parameters[0].theta == 7.0


def test_prior_initialization_uses_standardized_free_values(monkeypatch):
    from pysip.params.prior import Normal

    monkeypatch.setattr(Normal, "random", lambda self, hpd: np.array([2.0]))
    parameters = Parameters([
        {"name": "fixed", "value": 7.0, "transform": "fixed", "prior": Normal(0, 1)},
        {"name": "free", "value": 1.0, "loc": 10.0, "scale": 3.0,
         "transform": "log", "prior": Normal(0, 1)},
        {"name": "no_prior", "value": -4.0},
    ])
    initial = parameters.init_parameters(1, "prior")
    np.testing.assert_allclose(initial, [np.log(2.0), -4.0])
    np.testing.assert_allclose(parameters.theta, [7.0, 16.0, -4.0])
    assert parameters.prior == pytest.approx(Normal(0, 1).logpdf(2.0))


def test_value_initialization_uses_original_values_and_absolute_radius(monkeypatch):
    parameters = Parameters([
        {"name": "negative", "value": -4.0},
        {"name": "zero", "value": 0.0},
        {"name": "positive", "value": 4.0},
    ])
    monkeypatch.setattr(np.random, "uniform", lambda low, high: low + 0.8 * (high - low))
    initial = parameters.init_parameters(3, "value")
    np.testing.assert_allclose(initial, np.array([[-3.4], [0.0], [4.6]]) @ np.ones((1, 3)))


def test_value_initialization_respects_open_bounds(monkeypatch):
    parameters = Parameters([
        {"name": "lower", "value": -4.9, "bounds": (-5.0, None)},
        {"name": "upper", "value": 4.9, "bounds": (None, 5.0)},
        {"name": "interval", "value": 0.99, "bounds": (0.0, 1.0)},
    ])
    monkeypatch.setattr(np.random, "uniform", lambda low, high: (low + high) / 2)
    initial = parameters.init_parameters(2, "value")
    assert np.isfinite(initial).all()
    assert all(p.transform.in_bounds(p.theta_sd) for p in parameters.parameters)


def test_same_name_composition_preserves_both_groups():
    left = Parameters([{"name": "a", "value": 1.0}], name="same")
    right = Parameters([{"name": "a", "value": 2.0}], name="same")
    combined = left + right
    assert combined.name == "same__same"
    assert combined.ids == ["same__left__a", "same__right__a"]
    combined.theta_sd = [3.0, 4.0]
    np.testing.assert_allclose(left.theta, [3.0])
    np.testing.assert_allclose(right.theta, [4.0])
    combined.set_parameter("same__left", "a", value=5.0)
    np.testing.assert_allclose(combined.theta, [5.0, 4.0])
    nested = combined + (left + right)
    assert len(nested.ids) == len(set(nested.ids)) == 4


@pytest.mark.parametrize("operation", ["sum", "product"])
def test_same_name_kernel_composition_preserves_parameter_vectors(operation):
    from pysip.statespace import Matern32

    left, right = Matern32(), Matern32()
    composed = left + right if operation == "sum" else left * right
    parameters = composed.parameters
    assert parameters.n_par == 6
    assert len(set(parameters.ids)) == 6
    assert parameters.ids[:3] == [
        f"Matern32__left__{name}" for name in ("mscale", "lscale", "sigv")
    ]
    assert parameters.ids[3:] == [
        f"Matern32__right__{name}" for name in ("mscale", "lscale", "sigv")
    ]
    assert parameters.ids_free == [
        id_ for id_, free in zip(parameters.ids, parameters.free) if free
    ]
    parameters.theta_free = np.arange(1.0, parameters.n_par_free + 1.0)
    composed.update()
    assert np.isfinite(composed.A).all()
