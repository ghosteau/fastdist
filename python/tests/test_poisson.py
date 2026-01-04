# tests/python_modules/test_poisson.py
from unittest.mock import patch

import pytest

import fastdist.distributions.poisson as poisson_module
from fastdist.distributions.poisson import Poisson


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
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


@pytest.mark.parametrize("lambda_", [0, -1, -0.5])
def test_validate_params_rejects_invalid_values(lambda_):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
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
    mock_core.poisson_variance.return_value = 2.0

    p = Poisson(lambda_=2.0)
    result = p.variance()

    mock_core.poisson_variance.assert_called_once_with(2.0)
    assert result == 2.0


def test_stddev_delegates_to_core(mock_core):
    mock_core.poisson_stddev.return_value = 1.414

    p = Poisson(lambda_=2.0)
    result = p.stddev()

    mock_core.poisson_stddev.assert_called_once_with(2.0)
    assert result == 1.414


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.poisson_mgf_scalar.return_value = 7.389

    p = Poisson(lambda_=2.0)
    result = p.mgf_scalar(0.5)

    mock_core.poisson_mgf_scalar.assert_called_once_with(0.5, 2.0)
    assert result == 7.389


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.poisson_cgf_scalar.return_value = 2.297

    p = Poisson(lambda_=2.0)
    result = p.cgf_scalar(0.5)

    mock_core.poisson_cgf_scalar.assert_called_once_with(0.5, 2.0)
    assert result == 2.297


def test_sample_delegates_to_core(mock_core):
    mock_core.poisson_sample.return_value = 3

    p = Poisson(lambda_=2.0)
    result = p.sample()

    mock_core.poisson_sample.assert_called_once_with(2.0)
    assert result == 3


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
        (Poisson._mean, (0,)),
        (Poisson._mean, (-1,)),
        (Poisson._mean, (-0.5,)),
        (Poisson._variance, (0,)),
        (Poisson._variance, (-1,)),
        (Poisson._variance, (-0.5,)),
        (Poisson._stddev, (0,)),
        (Poisson._stddev, (-1,)),
        (Poisson._stddev, (-0.5,)),
        (Poisson._mgf_scalar, (0.5, 0)),
        (Poisson._mgf_scalar, (0.5, -1)),
        (Poisson._cgf_scalar, (0.5, 0)),
        (Poisson._cgf_scalar, (0.5, -0.5)),
        (Poisson._sample, (0,)),
        (Poisson._sample, (-1,)),
    ],
)
def test_classmethods_reject_invalid_parameters(method, args):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pmf_scalar", "poisson_pmf_scalar", (3, 2.0), 0.18),
    ("_cdf_scalar", "poisson_cdf_scalar", (3, 2.0), 0.857),
    ("_mean", "poisson_mean", (2.0,), 2.0),
    ("_variance", "poisson_variance", (2.0,), 2.0),
    ("_stddev", "poisson_stddev", (2.0,), 1.414),
    ("_mgf_scalar", "poisson_mgf_scalar", (0.5, 2.0), 7.389),
    ("_cgf_scalar", "poisson_cgf_scalar", (0.5, 2.0), 2.297),
    ("_sample", "poisson_sample", (2.0,), 3),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(Poisson, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    p = Poisson(lambda_=2.0)
    with pytest.raises(AttributeError):
        p.extra = 123