# tests/python_modules/test_exponential.py
from unittest.mock import patch

import pytest

import fastdist.distributions.exponential as exp_module
from fastdist.distributions.exponential import Exponential


@pytest.fixture
def mock_core():
    with patch.object(exp_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    e = Exponential(lambda_=0.5)
    assert e.lambda_ == 0.5


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
# Class method delegation to core
# ---------------------------------------------------------------------------

def test_pdf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_pdf_scalar.return_value = 0.3679
    result = Exponential.pdf_scalar(1.0, 1.0)
    mock_core.exponential_pdf_scalar.assert_called_once_with(1.0, 1.0)
    assert result == 0.3679


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_cdf_scalar.return_value = 0.6321
    result = Exponential.cdf_scalar(1.0, 1.0)
    mock_core.exponential_cdf_scalar.assert_called_once_with(1.0, 1.0)
    assert result == 0.6321


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_mgf_scalar.return_value = 2.5
    result = Exponential.mgf_scalar(0.5, 2.0)
    mock_core.exponential_mgf_scalar.assert_called_once_with(0.5, 2.0)
    assert result == 2.5


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_cgf_scalar.return_value = 0.916
    result = Exponential.cgf_scalar(0.5, 2.0)
    mock_core.exponential_cgf_scalar.assert_called_once_with(0.5, 2.0)
    assert result == 0.916


# ---------------------------------------------------------------------------
# Instance methods that rely on self.lambda_
# ---------------------------------------------------------------------------

def test_instance_methods_delegation(mock_core):
    mock_core.exponential_sample.return_value = 0.347
    mock_core.exponential_mean.return_value = 2.0
    mock_core.exponential_variance.return_value = 4.0
    mock_core.exponential_stddev.return_value = 2.0

    e = Exponential(lambda_=2.0)

    # sample
    result = e.sample()
    mock_core.exponential_sample.assert_called_once_with(2.0)
    assert result == 0.347

    # mean
    result = e.mean()
    mock_core.exponential_mean.assert_called_once_with(2.0)
    assert result == 2.0

    # variance
    result = e.variance()
    mock_core.exponential_variance.assert_called_once_with(2.0)
    assert result == 4.0

    # stddev
    result = e.stddev()
    mock_core.exponential_stddev.assert_called_once_with(2.0)
    assert result == 2.0


# ---------------------------------------------------------------------------
# Reject invalid lambda_ in instance methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method,invalid_lambda",
    [
        (Exponential.mean, 0),
        (Exponential.variance, -0.5),
        (Exponential.stddev, -2),
        (Exponential.sample, 0),
        (Exponential.sample, -1),
    ]
)
def test_instance_methods_reject_invalid_lambda(method, invalid_lambda):
    e = Exponential(lambda_=1.0)
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(e, invalid_lambda)  # pass invalid lambda_ explicitly


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    e = Exponential(lambda_=0.5)
    with pytest.raises(AttributeError):
        e.extra = 123
