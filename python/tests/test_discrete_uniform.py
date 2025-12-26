from unittest.mock import patch

import pytest

import fastdist.distributions.discrete_uniform as du_module
from fastdist.distributions.discrete_uniform import DiscreteUniform


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(du_module, "_core", autospec=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    du = DiscreteUniform(a=1, b=10)
    assert du.a == 1
    assert du.b == 10


def test_init_allows_none_parameters():
    du = DiscreteUniform()
    assert du.a is None
    assert du.b is None


@pytest.mark.parametrize(
    "a,b",
    [
        (5, 5),  # equal
        (6, 5),  # a > b
        (10, 1),  # a > b
    ]
)
def test_init_invalid_range_raises(a, b):
    with pytest.raises(ValueError, match="a must be less than b"):
        DiscreteUniform(a=a, b=b)


def test_repr():
    du = DiscreteUniform(a=3, b=8)
    # reflects current implementation
    assert repr(du) == "DiscreteUniform(mu=3, sigma=8)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

def test_validate_params_accepts_valid_ranges():
    DiscreteUniform._validate_params(a=1, b=2)
    DiscreteUniform._validate_params(a=-10, b=-5)
    DiscreteUniform._validate_params(a=None, b=None)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pmf_delegates_to_core(mock_core):
    mock_core.discrete_uniform_pmf_scalar.return_value = 0.1

    du = DiscreteUniform(a=1, b=5)
    result = du.pmf(3)

    mock_core.discrete_uniform_pmf_scalar.assert_called_once_with(3, 1, 5)
    assert result == 0.1


def test_cdf_delegates_to_core(mock_core):
    mock_core.discrete_uniform_cdf_scalar.return_value = 0.7

    du = DiscreteUniform(a=1, b=5)
    result = du.cdf(3)

    mock_core.discrete_uniform_cdf_scalar.assert_called_once_with(3, 1, 5)
    assert result == 0.7


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_mean_delegates_to_core(mock_core):
    mock_core.discrete_uniform_mean.return_value = 2.5

    du = DiscreteUniform(a=1, b=4)
    result = du.mean()

    mock_core.discrete_uniform_mean.assert_called_once_with(1, 4)
    assert result == 2.5


def test_variance_delegates_to_core(mock_core):
    mock_core.discrete_uniform_variance.return_value = 1.25

    du = DiscreteUniform(a=1, b=4)
    result = du.variance()

    mock_core.discrete_uniform_variance.assert_called_once_with(1, 4)
    assert result == 1.25


def test_stddev_delegates_to_core(mock_core):
    mock_core.discrete_uniform_stddev.return_value = 1.118

    du = DiscreteUniform(a=1, b=4)
    result = du.stddev()

    mock_core.discrete_uniform_stddev.assert_called_once_with(1, 4)
    assert result == 1.118


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (DiscreteUniform._pmf_scalar, (3, 5, 2)),
        (DiscreteUniform._cdf_scalar, (3, 6, 1)),
        (DiscreteUniform._mean, (10, 1)),
        (DiscreteUniform._variance, (5, 4)),
        (DiscreteUniform._stddev, (7, 3)),
    ]
)
def test_classmethods_reject_invalid_range(method, args):
    with pytest.raises(ValueError, match="a must be less than b"):
        method(*args)


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    du = DiscreteUniform(a=1, b=5)
    with pytest.raises(AttributeError):
        du.extra = 123
