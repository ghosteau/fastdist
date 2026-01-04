# tests/python_modules/test_exponential.py
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


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_cdf_scalar.return_value = 0.6321
    e = Exponential(lambda_=1.0)
    result = e.cdf_scalar(1.0)
    mock_core.exponential_cdf_scalar.assert_called_once_with(1.0, 1.0)
    assert result == 0.6321


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_mgf_scalar.return_value = 2.5
    e = Exponential(lambda_=2.0)
    result = e.mgf_scalar(0.5)
    mock_core.exponential_mgf_scalar.assert_called_once_with(0.5, 2.0)
    assert result == 2.5


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.exponential_cgf_scalar.return_value = 0.916
    e = Exponential(lambda_=2.0)
    result = e.cgf_scalar(0.5)
    mock_core.exponential_cgf_scalar.assert_called_once_with(0.5, 2.0)
    assert result == 0.916


def test_sample_delegates_to_core(mock_core):
    mock_core.exponential_sample.return_value = 0.347
    e = Exponential(lambda_=2.0)
    result = e.sample()
    mock_core.exponential_sample.assert_called_once_with(2.0)
    assert result == 0.347


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
        (Exponential._sample, 0),
        (Exponential._sample, -1),
    ]
)
def test_classmethods_reject_invalid_lambda_scalar_only(method, lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(lambda_)


# Methods that require t as argument
@pytest.mark.parametrize(
    "method,args",
    [
        (Exponential._mgf_scalar, (0.5, 0)),
        (Exponential._mgf_scalar, (0.5, -1)),
        (Exponential._cgf_scalar, (0.5, 0)),
        (Exponential._cgf_scalar, (0.5, -0.5)),
    ]
)
def test_classmethods_reject_invalid_lambda_with_t(method, args):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pdf_scalar", "exponential_pdf_scalar", (1.0, 1.0), 0.3679),
    ("_cdf_scalar", "exponential_cdf_scalar", (1.0, 1.0), 0.6321),
    ("_mean", "exponential_mean", (0.5,), 2.0),
    ("_variance", "exponential_variance", (0.5,), 4.0),
    ("_stddev", "exponential_stddev", (0.5,), 2.0),
    ("_mgf_scalar", "exponential_mgf_scalar", (0.5, 2.0), 2.5),
    ("_cgf_scalar", "exponential_cgf_scalar", (0.5, 2.0), 0.916),
    ("_sample", "exponential_sample", (2.0,), 0.347),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(Exponential, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    e = Exponential(lambda_=0.5)
    with pytest.raises(AttributeError):
        e.extra = 123
