import importlib

import pytest

import probforge
import probforge.distributions as distributions

EXPECTED = [
    "Bernoulli", "Beta", "Binomial", "ChiSquare",
    "DiscreteUniform", "Exponential", "Gamma",
    "Geometric", "Hypergeometric", "NegativeBinomial",
    "Normal", "Poisson", "Uniform", "Utils",
]

# Module-level functions, as opposed to the distribution classes. These live at
# the package root because they act on the one engine every sampler shares, so
# they belong to no single distribution.
EXPECTED_FUNCTIONS = ["seed", "seed_from_entropy"]


def test_top_level_all_matches_expected():
    # the top level exports every distribution class, the module-level
    # functions, and __version__
    assert sorted(probforge.__all__) == sorted(EXPECTED + EXPECTED_FUNCTIONS + ["__version__"])


@pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
def test_every_exported_function_is_callable(name):
    assert callable(getattr(probforge, name)), f"{name} is not callable"


def test_distributions_all_matches_top_level():
    # the distributions subpackage exports the classes only
    assert sorted(distributions.__all__) == sorted(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_every_exported_name_is_importable(name):
    assert hasattr(probforge, name), f"{name} missing from probforge"
    assert hasattr(distributions, name), f"{name} missing from probforge.distributions"


@pytest.mark.parametrize("name", EXPECTED)
def test_every_class_has_a_module(name):
    # every exported class must come from a real submodule, not a stale alias
    cls = getattr(probforge, name)
    assert importlib.import_module(cls.__module__) is not None