import pytest

from pysip.statespace import GPProduct, Matern12, Matern32, Matern52


@pytest.fixture
def matern12():
    return Matern12(name="m12")


@pytest.fixture
def matern32():
    return Matern32()


@pytest.fixture
def matern52():
    return Matern52(name="m52")


@pytest.fixture
def product(matern12, matern32):
    product = GPProduct(matern12, matern32)
    product.parameters.theta = list(range(1, 7))
    product.update()
    return product


@pytest.fixture
def product_double(product, matern52):
    product_double = product * matern52
    product_double.parameters.theta = list(range(1, 10))
    product_double.update()
    return product_double


matern_params = ["mscale", "lscale", "sigv"]


def suffix(s):
    return [s + "__" + x for x in matern_params]


def test_names(product, matern12, matern32):
    assert product.parameters.ids == suffix(matern12.name) + suffix(matern32.name)
    assert list(product.parameters._parameters.keys()) == ["m12", "Matern32"]
    assert list(product.parameters._parameters["m12"].keys()) == matern_params
    assert list(product.parameters._parameters["Matern32"].keys()) == matern_params


def test_names_double(product_double, product, matern12, matern32, matern52):
    assert product_double.parameters.ids == (
        suffix(product.name + "__" + matern12.name)
        + suffix(product.name + "__" + matern32.name)
        + suffix(matern52.name)
    )
    par = product_double.parameters._parameters
    assert list(par.keys()) == ["m12__x__Matern32", "m52"]
    assert list(par["m12__x__Matern32"].keys()) == ["m12", "Matern32"]
    assert list(par["m52"].keys()) == matern_params
    assert list(par["m12__x__Matern32"]["m12"].keys()) == matern_params
    assert list(par["m12__x__Matern32"]["Matern32"].keys()) == matern_params


def test_same_name_product_preserves_both_parameter_groups():
    left, right = Matern32(), Matern32()
    combined = left * right
    assert combined.name == "Matern32__x__Matern32"
    assert combined.parameters.ids == suffix("Matern32__left") + suffix("Matern32__right")
    assert [node.name for node in combined.params] == combined.parameters.ids
    assert combined.parameters.n_par == 6
    combined.parameters.set_parameter("Matern32__left__lscale", value=2.0)
    combined.parameters.set_parameter("Matern32__right__lscale", value=3.0)
    combined.update()
    assert combined._gp1.parameters.theta[1] == 2.0
    assert combined._gp2.parameters.theta[1] == 3.0
    nested = combined * combined
    nested.update()
    for index in (3, 5, 6, 8, 9, 11):
        assert not nested.parameters.free[index]
    assert nested.parameters.theta[6] == 1.0
    assert nested.parameters.theta[8] == 0.0
    assert len(set(nested.parameters.ids)) == 12
    assert [node.name for node in nested.params] == nested.parameters.ids
    assert left.parameters.ids == matern_params
    assert right.parameters.ids == matern_params
