from unittest.mock import patch

import pytest

import fastdist.distributions.poisson as poisson_module
from fastdist.distributions.poisson import Poisson


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(poisson_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    p = Poisson(lambda_=3.5)
    assert p.lambda_ == 3.5


def test_init_allows_none_parameter():
    p = Poisson()
    assert p.lambda_ is None


@pytest.mark.parametrize("lambda_", [0, -1, -0.5])
def test_init_invalid_lambda_raises(lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        Poisson(lambda_=lambda_)


def test_repr():
    p = Poisson(lambda_=2.0)
    assert repr(p) == "Poisson(lambda_=2.0)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lambda_", [0.0001, 1, 10])
def test_validate_params_accepts_valid_values(lambda_):
    Poisson._validate_params(lambda_=lambda_)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pmf_scalar_delegates_to_core(mock_core):
    mock_core.poisson_pmf_scalar.return_value = 0.2
    p = Poisson(lambda_=2.0)
    result = p.pmf_scalar(3)
    mock_core.poisson_pmf_scalar.assert_called_once_with(3, 2.0)
    assert result == 0.2


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.poisson_cdf_scalar.return_value = 0.8
    p = Poisson(lambda_=2.0)
    result = p.cdf_scalar(3)
    mock_core.poisson_cdf_scalar.assert_called_once_with(3, 2.0)
    assert result == 0.8


def test_mean_delegates_to_core(mock_core):
    mock_core.poisson_mean.return_value = 2.0
    p = Poisson(lambda_=2.0)
    result = p.mean()
    mock_core.poisson_mean.assert_called_once_with(2.0)
    assert result == 2.0


def test_variance_delegates_to_core(mock_core):
    mock_core.poisson_variance.return_value = 1.5
    p = Poisson(lambda_=2.0)
    result = p.variance()
    mock_core.poisson_variance.assert_called_once_with(2.0)
    assert result == 1.5


def test_stddev_delegates_to_core(mock_core):
    mock_core.poisson_stddev.return_value = 1.2247
    p = Poisson(lambda_=2.0)
    result = p.stddev()
    mock_core.poisson_stddev.assert_called_once_with(2.0)
    assert result == 1.2247


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (Poisson._pmf_scalar, (3, 0)),
        (Poisson._pmf_scalar, (3, -1)),
        (Poisson._pmf_scalar, (3, -0.5)),
        (Poisson._cdf_scalar, (3, 0)),
        (Poisson._cdf_scalar, (3, -1)),
        (Poisson._cdf_scalar, (3, -0.5)),
    ]
)
def test_classmethods_reject_invalid_lambda_with_k(method, args):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(*args)


@pytest.mark.parametrize(
    "method, lambda_",
    [
        (Poisson._mean, 0),
        (Poisson._mean, -1),
        (Poisson._mean, -0.5),
        (Poisson._variance, 0),
        (Poisson._variance, -1),
        (Poisson._variance, -0.5),
        (Poisson._stddev, 0),
        (Poisson._stddev, -1),
        (Poisson._stddev, -0.5),
    ]
)
def test_classmethods_reject_invalid_lambda_scalar_only(method, lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(lambda_)
