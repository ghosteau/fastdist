import importlib

import pytest

import fastdist
import fastdist.distributions as distributions

EXPECTED = [
    "Bernoulli", "Beta", "Binomial", "ChiSquare",
    "DiscreteUniform", "Exponential", "Gamma",
    "Geometric", "NegativeBinomial", "Normal",
    "Poisson", "Uniform", "Utils",
]


def test_top_level_all_matches_expected():
    assert sorted(fastdist.__all__) == sorted(EXPECTED)


def test_distributions_all_matches_top_level():
    assert sorted(distributions.__all__) == sorted(fastdist.__all__)


@pytest.mark.parametrize("name", EXPECTED)
def test_every_exported_name_is_importable(name):
    assert hasattr(fastdist, name), f"{name} missing from fastdist"
    assert hasattr(distributions, name), f"{name} missing from fastdist.distributions"


@pytest.mark.parametrize("name", EXPECTED)
def test_every_class_has_a_module(name):
    # every exported class must come from a real submodule, not a stale alias
    cls = getattr(fastdist, name)
    assert importlib.import_module(cls.__module__) is not None