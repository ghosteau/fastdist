from unittest.mock import patch

import pytest

import fastdist.distributions.geometric as geo_module
from fastdist.distributions.geometric import Geometric


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(geo_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    g = Geometric(p=0.5)
    assert g.p == 0.5


def test_init_allows_none_parameter():
    g = Geometric()
    assert g.p is None


@pytest.mark.parametrize("p", [0, -0.1, 1.5])
def test_init_invalid_p_raises(p):
    with pytest.raises(ValueError, match="p must be in the interval"):
        Geometric(p=p)


def test_repr():
    g = Geometric(p=0.3)
    assert repr(g) == "Geometric(p=0.3)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.0001, 0.5, 1.0])
def test_validate_params_accepts_valid_values(p):
    Geometric._validate_params(p=p)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pmf_scalar_delegates_to_core(mock_core):
    mock_core.geometric_pmf_scalar.return_value = 0.25
    g = Geometric(p=0.5)
    result = g.pmf_scalar(2)
    mock_core.geometric_pmf_scalar.assert_called_once_with(2, 0.5)
    assert result == 0.25


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.geometric_cdf_scalar.return_value = 0.75
    g = Geometric(p=0.5)
    result = g.cdf_scalar(2)
    mock_core.geometric_cdf_scalar.assert_called_once_with(2, 0.5)
    assert result == 0.75


def test_mean_delegates_to_core(mock_core):
    mock_core.geometric_mean.return_value = 2.0
    g = Geometric(p=0.5)
    result = g.mean()
    mock_core.geometric_mean.assert_called_once_with(0.5)
    assert result == 2.0


def test_variance_delegates_to_core(mock_core):
    mock_core.geometric_variance.return_value = 1.5
    g = Geometric(p=0.5)
    result = g.variance()
    mock_core.geometric_variance.assert_called_once_with(0.5)
    assert result == 1.5


def test_stddev_delegates_to_core(mock_core):
    mock_core.geometric_stddev.return_value = 1.2247
    g = Geometric(p=0.5)
    result = g.stddev()
    mock_core.geometric_stddev.assert_called_once_with(0.5)
    assert result == 1.2247


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (Geometric._pmf_scalar, (2, 0)),
        (Geometric._pmf_scalar, (2, -0.1)),
        (Geometric._pmf_scalar, (2, 1.5)),
        (Geometric._cdf_scalar, (2, 0)),
        (Geometric._cdf_scalar, (2, -0.1)),
        (Geometric._cdf_scalar, (2, 1.5)),
    ]
)
def test_classmethods_reject_invalid_p_with_k(method, args):
    with pytest.raises(ValueError, match="p must be in the interval"):
        method(*args)


@pytest.mark.parametrize(
    "method, p",
    [
        (Geometric._mean, 0),
        (Geometric._mean, -0.1),
        (Geometric._mean, 1.5),
        (Geometric._variance, 0),
        (Geometric._variance, -0.1),
        (Geometric._variance, 1.5),
        (Geometric._stddev, 0),
        (Geometric._stddev, -0.1),
        (Geometric._stddev, 1.5),
    ]
)
def test_classmethods_reject_invalid_p_scalar_only(method, p):
    with pytest.raises(ValueError, match="p must be in the interval"):
        method(p)
