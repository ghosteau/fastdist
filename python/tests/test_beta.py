# tests/python_modules/test_beta.py
from unittest.mock import patch

import pytest

import fastdist.distributions.beta as beta_module
from fastdist.distributions.beta import Beta


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
    with patch.object(beta_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    b = Beta(alpha=2.0, beta=3.0)
    assert b.alpha == 2.0
    assert b.beta == 3.0


def test_init_allows_none_parameters():
    b = Beta()
    assert b.alpha is None
    assert b.beta is None


@pytest.mark.parametrize("alpha,beta", [
    (-0.1, 1.0),
    (0, 1.0),
    (-1, 1.0),
    (1.0, -0.1),
    (1.0, 0),
    (1.0, -1),
    (-1, -1),
])
def test_init_invalid_parameters_raises(alpha, beta):
    with pytest.raises(ValueError, match="must be positive"):
        Beta(alpha=alpha, beta=beta)


def test_repr():
    b = Beta(alpha=2.0, beta=3.0)
    assert repr(b) == "Beta(alpha=2.0, beta=3.0)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,beta", [
    (0.1, 0.1),
    (1.0, 1.0),
    (2.5, 3.5),
    (100, 200),
])
def test_validate_params_accepts_valid_values(alpha, beta):
    Beta._validate_params(alpha=alpha, beta=beta)


@pytest.mark.parametrize("alpha,beta", [
    (0, 1.0),
    (-1, 1.0),
    (1.0, 0),
    (1.0, -1),
])
def test_validate_params_rejects_invalid_values(alpha, beta):
    with pytest.raises(ValueError, match="must be positive"):
        Beta._validate_params(alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, value", [
    ("pdf_scalar", "beta_pdf_scalar", 0.5),
    ("cdf_scalar", "beta_cdf_scalar", 0.3),
    ("mean", "beta_mean", 0.4),
    ("variance", "beta_variance", 0.048),
    ("stddev", "beta_stddev", 0.219),
    ("sample", "beta_sample", 0.42),
])
def test_instance_methods_delegate_to_core(mock_core, method_name, core_method_name, value):
    getattr(mock_core, core_method_name).return_value = value
    b = Beta(alpha=2.0, beta=3.0)
    method = getattr(b, method_name)
    if "scalar" in method_name:
        result = method(0.5)
        getattr(mock_core, core_method_name).assert_called_once_with(0.5, 2.0, 3.0)
    else:
        result = method()
        getattr(mock_core, core_method_name).assert_called_once_with(2.0, 3.0)
    assert result == value


# ---------------------------------------------------------------------------
# Classmethod validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, args", [
    ("_pdf_scalar", (0.5, -0.1, 1.0)),
    ("_pdf_scalar", (0.5, 1.0, -0.1)),
    ("_pdf_scalar", (0.5, 0, 1.0)),
    ("_pdf_scalar", (0.5, 1.0, 0)),
    ("_cdf_scalar", (0.5, -1, 1.0)),
    ("_cdf_scalar", (0.5, 1.0, -1)),
    ("_mean", (-0.5, 1.0)),
    ("_mean", (1.0, -0.5)),
    ("_variance", (0, 1.0)),
    ("_variance", (1.0, 0)),
    ("_stddev", (-1, 1.0)),
    ("_stddev", (1.0, -1)),
    ("_sample", (-0.1, 1.0)),
    ("_sample", (1.0, -0.1)),
])
def test_classmethods_reject_invalid_parameters(method_name, args):
    method = getattr(Beta, method_name)
    with pytest.raises(ValueError, match=r"must be positive"):
        method.__func__(Beta, *args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pdf_scalar", "beta_pdf_scalar", (0.5, 2.0, 3.0), 1.5),
    ("_cdf_scalar", "beta_cdf_scalar", (0.5, 2.0, 3.0), 0.6875),
    ("_mean", "beta_mean", (2.0, 3.0), 0.4),
    ("_variance", "beta_variance", (2.0, 3.0), 0.048),
    ("_stddev", "beta_stddev", (2.0, 3.0), 0.219),
    ("_sample", "beta_sample", (2.0, 3.0), 0.42),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(Beta, method_name)
    result = method.__func__(Beta, *args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value
