# tests/python_modules/test_bernoulli.py
from unittest.mock import patch

import pytest

import fastdist.distributions.bernoulli as bern_module
from fastdist.distributions.bernoulli import Bernoulli


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
    with patch.object(bern_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    b = Bernoulli(p=0.3)
    assert b.p == 0.3


@pytest.mark.parametrize("p", [-0.1, -1, 1.1, 2])
def test_init_invalid_p_raises(p):
    with pytest.raises(ValueError, match="p must be in the interval \\[0, 1\\]"):
        Bernoulli(p=p)


def test_repr():
    b = Bernoulli(p=0.7)
    assert repr(b) == "Bernoulli(p=0.7)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0, 0.5, 1])
def test_validate_params_accepts_valid_values(p):
    Bernoulli._validate_params(p=p)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, value", [
    ("pmf_scalar", "bernoulli_pmf_scalar", 1.0),
    ("cdf_scalar", "bernoulli_cdf_scalar", 1.0),
    ("mean", "bernoulli_mean", 0.5),
    ("variance", "bernoulli_variance", 0.25),
    ("stddev", "bernoulli_stddev", 0.5),
    ("mgf_scalar", "bernoulli_mgf", 1.2),
    ("cgf_scalar", "bernoulli_cgf", 0.182),
])
def test_instance_methods_delegate_to_core(mock_core, method_name, core_method_name, value):
    getattr(mock_core, core_method_name).return_value = value
    b = Bernoulli(p=0.5)
    method = getattr(b, method_name)
    if "scalar" in method_name:
        result = method(1)
        getattr(mock_core, core_method_name).assert_called_once_with(1, 0.5)
    else:
        result = method()
        getattr(mock_core, core_method_name).assert_called_once_with(0.5)
    assert result == value


# ---------------------------------------------------------------------------
# Classmethod validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1.0, -0.1)),
    ("_cdf_scalar", (1.0, 1.1)),
    ("_mean", (2,)),
    ("_variance", (1.5,)),
    ("_stddev", (-0.5,)),
    ("_mgf_scalar", (1.0, -0.1)),
    ("_cgf_scalar", (1.0, 1.5)),
])
def test_classmethods_reject_invalid_p(method_name, args):
    method = getattr(Bernoulli, method_name)
    with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
        method.__func__(Bernoulli, *args)
