import pickle
from copy import copy, deepcopy

import pytest

from pysip.statespace import Models
from pysip.utils.misc import Namespace


def test_namespace_mapping_and_attribute_mutation():
    ns = Namespace(first=1)
    ns.second = 2
    ns.update([("third", 3)], fourth=4)
    ns.setdefault("fifth", 5)
    assert ns == {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    assert ns.third == 3
    ns["first"] = 10
    assert ns.first == 10
    del ns.second
    assert "second" not in ns
    assert ns.pop("third") == 3
    with pytest.raises(AttributeError):
        _ = ns.third
    with pytest.raises(AttributeError):
        del ns.missing
    assert ns.popitem() == ("fifth", 5)
    ns.clear()
    assert not ns
    with pytest.raises(AttributeError):
        _ = ns.first


@pytest.mark.parametrize("clone", [copy, deepcopy, lambda ns: pickle.loads(pickle.dumps(ns))])
def test_namespace_copy_and_pickle_keep_attribute_mapping_coherent(clone):
    original = Namespace(value=[1])
    restored = clone(original)
    assert isinstance(restored, Namespace)
    assert restored == original
    restored["value"] = [2]
    assert restored.value == [2]
    restored.value = [3]
    assert restored["value"] == [3]
    assert original.value == [1]


def test_model_registry_attribute_api():
    assert Models.Matern32 is Models["Matern32"]
