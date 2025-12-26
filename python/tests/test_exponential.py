from unittest.mock import patch

import pytest

import fastdist.distributions.exponential as exp_module
from fastdist.distributions.exponential import Exponential


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(exp_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    e = Exponential(lambda_=0.5)
    assert e.lambda_ == 0.5


def test_init_allows_none_parameter():
    e = Exponential()
    assert e.lambda_ is None


@pytest.mark.parametrize("lambda_", [0, -1, -0.5])
def test_init_invalid_lambda_raises(lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        Exponential(lambda_=lambda_)


def test_repr():
    e = Exponential(lambda_=0.7)
    assert repr(e) == "Exponential(lambda_=0.7)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lambda_", [0.0001, 1, 10])
def test_validate_params_accepts_valid_values(lambda_):
    Exponential._validate_params(lambda_=lambda_)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pdf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_pdf_scalar.return_value = 0.3679
    e = Exponential(lambda_=1.0)
    result = e.pdf_scalar(1.0)
    mock_core.exponential_pdf_scalar.assert_called_once_with(1.0, 1.0)
    assert result == 0.3679


def test_logpdf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_logpdf_scalar.return_value = -1.0
    e = Exponential(lambda_=1.0)
    result = e.logpdf_scalar(1.0)
    mock_core.exponential_logpdf_scalar.assert_called_once_with(1.0, 1.0)
    assert result == -1.0


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_cdf_scalar.return_value = 0.6321
    e = Exponential(lambda_=1.0)
    result = e.cdf_scalar(1.0)
    mock_core.exponential_cdf_scalar.assert_called_once_with(1.0, 1.0)
    assert result == 0.6321


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_mean_delegates_to_core(mock_core):
    mock_core.exponential_mean.return_value = 2.0
    e = Exponential(lambda_=0.5)
    result = e.mean()
    mock_core.exponential_mean.assert_called_once_with(0.5)
    assert result == 2.0


def test_variance_delegates_to_core(mock_core):
    mock_core.exponential_variance.return_value = 4.0
    e = Exponential(lambda_=0.5)
    result = e.variance()
    mock_core.exponential_variance.assert_called_once_with(0.5)
    assert result == 4.0


def test_stddev_delegates_to_core(mock_core):
    mock_core.exponential_stddev.return_value = 2.0
    e = Exponential(lambda_=0.5)
    result = e.stddev()
    mock_core.exponential_stddev.assert_called_once_with(0.5)
    assert result == 2.0


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

# Methods that require x as argument
@pytest.mark.parametrize(
    "method,args",
    [
        (Exponential._pdf_scalar, (1.0, 0)),
        (Exponential._logpdf_scalar, (1.0, -1)),
        (Exponential._cdf_scalar, (1.0, 0)),
    ]
)
def test_classmethods_reject_invalid_lambda_with_x(method, args):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(*args)


# Methods that only take lambda_
@pytest.mark.parametrize(
    "method,lambda_",
    [
        (Exponential._mean, 0),
        (Exponential._variance, -0.5),
        (Exponential._stddev, -2),
    ]
)
def test_classmethods_reject_invalid_lambda_scalar_only(method, lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(lambda_)
