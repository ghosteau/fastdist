from unittest.mock import patch

import pytest

# Changed: Imported from uniform module
import fastdist.distributions.uniform as uniform_module
from fastdist.distributions.uniform import Uniform


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    # Changed: Patching the uniform module's core
    with patch.object(uniform_module, "_core", autospec=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    # Changed: Uses 'a' and 'b' parameters
    u = Uniform(a=0.0, b=1.0)
    assert u.a == 0.0
    assert u.b == 1.0


def test_init_allows_none_parameters():
    u = Uniform()
    assert u.a is None
    assert u.b is None


# Changed: Tests for invalid 'a' and 'b' relation (a >= b)
@pytest.mark.parametrize(
    "a, b",
    [
        (1.0, 1.0),  # a == b
        (2.0, 1.0),  # a > b
        (0.0, 0.0),
        (-1.0, -1.0),
    ],
)
def test_init_invalid_params_raises(a, b):
    with pytest.raises(ValueError, match="a must be less than b"):
        Uniform(a=a, b=b)


def test_repr():
    # Changed: Uses 'a' and 'b' in the repr string
    u = Uniform(a=1.5, b=2.5)
    assert repr(u) == "Uniform(a=1.5, b=2.5)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

# Changed: Tests for a >= b
@pytest.mark.parametrize(
    "a, b",
    [
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_validate_params_rejects_a_greater_equal_b(a, b):
    with pytest.raises(ValueError):
        Uniform._validate_params(a=a, b=b)


# Changed: Tests for a < b
def test_validate_params_accepts_a_less_than_b():
    Uniform._validate_params(a=1.0, b=2.0)  # should not raise


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pdf_scalar_delegates_to_core(mock_core):
    # Changed: Mock the uniform specific core function
    mock_core.uniform_pdf_scalar.return_value = 0.5  # Example for U(0, 2) at x=1

    u = Uniform(a=0.0, b=2.0)
    result = u.pdf_scalar(1.0)

    # Changed: Check for (x, a, b) call
    mock_core.uniform_pdf_scalar.assert_called_once_with(1.0, 0.0, 2.0)
    assert result == 0.5


# Note: Uniform does not have logpdf_scalar

def test_cdf_scalar_delegates_to_core(mock_core):
    # Changed: Mock the uniform specific core function
    mock_core.uniform_cdf_scalar.return_value = 0.5

    u = Uniform(a=0.0, b=2.0)
    result = u.cdf_scalar(1.0)

    # Changed: Check for (x, a, b) call
    mock_core.uniform_cdf_scalar.assert_called_once_with(1.0, 0.0, 2.0)
    assert result == 0.5


# Note: Uniform does not have z_score

# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_mean_delegates_to_core(mock_core):
    # Changed: Mock the uniform specific core function
    mock_core.uniform_mean.return_value = 1.5

    u = Uniform(a=1.0, b=2.0)
    result = u.mean()

    # Changed: Check for (a, b) call
    mock_core.uniform_mean.assert_called_once_with(1.0, 2.0)
    assert result == 1.5


def test_variance_delegates_to_core(mock_core):
    # Changed: Mock the uniform specific core function
    mock_core.uniform_variance.return_value = 0.083333333

    u = Uniform(a=1.0, b=2.0)
    result = u.variance()

    # Changed: Check for (a, b) call
    mock_core.uniform_variance.assert_called_once_with(1.0, 2.0)
    assert result == 0.083333333


def test_stddev_delegates_to_core(mock_core):
    # Changed: Mock the uniform specific core function
    mock_core.uniform_stddev.return_value = 0.288675135

    u = Uniform(a=1.0, b=2.0)
    result = u.stddev()

    # Changed: Check for (a, b) call
    mock_core.uniform_stddev.assert_called_once_with(1.0, 2.0)
    assert result == 0.288675135


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

# Changed: Parametrized test for classmethods with invalid (a >= b) inputs
@pytest.mark.parametrize(
    "method, args",
    [
        (Uniform._pdf_scalar, (0.0, 1.0, 1.0)),  # x, a, b (a=b)
        (Uniform._cdf_scalar, (0.0, 1.0, 1.0)),  # x, a, b (a=b)
        (Uniform._mean, (2.0, 1.0)),  # a, b (a>b)
        (Uniform._variance, (1.0, 1.0)),  # a, b (a=b)
        (Uniform._stddev, (2.0, 1.0)),  # a, b (a>b)
    ],
)
def test_classmethods_reject_invalid_ab(method, args):
    with pytest.raises(ValueError, match="a must be less than b"):
        method(*args)


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    # Changed: Initialize Uniform
    u = Uniform(a=0.0, b=1.0)
    with pytest.raises(AttributeError):
        u.foo = 123
