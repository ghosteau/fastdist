# tests/python_modules/test_negative_binomial.py
from unittest.mock import patch

import pytest

import fastdist.distributions.negative_binomial as nb_module
from fastdist.distributions.negative_binomial import NegativeBinomial


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
    with patch.object(nb_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    nb = NegativeBinomial(r=5, p=0.6)
    assert nb.r == 5
    assert nb.p == 0.6


@pytest.mark.parametrize("r,p", [
    (-1, 0.5),
    (0, 0.5),
    (-5, 0.5),
    (5, -0.1),
    (5, 1.1),
    (5, 2),
    (0, -0.5),
])
def test_init_invalid_parameters_raises(r, p):
    with pytest.raises(ValueError):
        NegativeBinomial(r=r, p=p)


def test_repr():
    nb = NegativeBinomial(r=3, p=0.7)
    assert repr(nb) == "NegativeBinomial(r=3, p=0.7)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r,p", [
    (1, 0.0),
    (5, 0.5),
    (10, 1.0),
    (5, 0.3),
])
def test_validate_params_accepts_valid_values(r, p):
    NegativeBinomial._validate_params(r=r, p=p)


@pytest.mark.parametrize("r,p", [
    (0, 0.5),
    (-1, 0.5),
    (5, -0.1),
    (5, 1.1),
])
def test_validate_params_rejects_invalid_values(r, p):
    with pytest.raises(ValueError):
        NegativeBinomial._validate_params(r=r, p=p)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pmf_scalar_delegates_to_core(mock_core):
    mock_core.negative_binomial_pmf_scalar.return_value = 0.186

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.pmf_scalar(3)

    mock_core.negative_binomial_pmf_scalar.assert_called_once_with(3, 5, 0.6)
    assert result == 0.186


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.negative_binomial_cdf_scalar.return_value = 0.663

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.cdf_scalar(3)

    mock_core.negative_binomial_cdf_scalar.assert_called_once_with(3, 5, 0.6)
    assert result == 0.663


def test_mean_delegates_to_core(mock_core):
    mock_core.negative_binomial_mean.return_value = 3.333

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.mean()

    mock_core.negative_binomial_mean.assert_called_once_with(5, 0.6)
    assert result == 3.333


def test_variance_delegates_to_core(mock_core):
    mock_core.negative_binomial_variance.return_value = 5.556

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.variance()

    mock_core.negative_binomial_variance.assert_called_once_with(5, 0.6)
    assert result == 5.556


def test_stddev_delegates_to_core(mock_core):
    mock_core.negative_binomial_stddev.return_value = 2.357

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.stddev()

    mock_core.negative_binomial_stddev.assert_called_once_with(5, 0.6)
    assert result == 2.357


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.negative_binomial_mgf_scalar.return_value = 1.845

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.mgf_scalar(0.1)

    mock_core.negative_binomial_mgf_scalar.assert_called_once_with(0.1, 5, 0.6)
    assert result == 1.845


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.negative_binomial_cgf_scalar.return_value = 0.612

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.cgf_scalar(0.1)

    mock_core.negative_binomial_cgf_scalar.assert_called_once_with(0.1, 5, 0.6)
    assert result == 0.612


def test_sample_delegates_to_core(mock_core):
    mock_core.negative_binomial_sample.return_value = 4

    nb = NegativeBinomial(r=5, p=0.6)
    result = nb.sample()

    mock_core.negative_binomial_sample.assert_called_once_with(5, 0.6)
    assert result == 4


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (NegativeBinomial._pmf_scalar, (3, 0, 0.5)),
        (NegativeBinomial._pmf_scalar, (3, -1, 0.5)),
        (NegativeBinomial._pmf_scalar, (3, 5, -0.1)),
        (NegativeBinomial._pmf_scalar, (3, 5, 1.1)),
        (NegativeBinomial._cdf_scalar, (3, 0, 0.5)),
        (NegativeBinomial._cdf_scalar, (3, 5, -0.5)),
        (NegativeBinomial._cdf_scalar, (3, 5, 1.5)),
        (NegativeBinomial._mean, (0, 0.5)),
        (NegativeBinomial._mean, (-1, 0.5)),
        (NegativeBinomial._mean, (5, -0.1)),
        (NegativeBinomial._mean, (5, 1.1)),
        (NegativeBinomial._variance, (0, 0.5)),
        (NegativeBinomial._variance, (5, -0.5)),
        (NegativeBinomial._stddev, (-1, 0.5)),
        (NegativeBinomial._stddev, (5, 1.5)),
        (NegativeBinomial._mgf_scalar, (0.1, 0, 0.5)),
        (NegativeBinomial._mgf_scalar, (0.1, 5, -0.1)),
        (NegativeBinomial._cgf_scalar, (0.1, -1, 0.5)),
        (NegativeBinomial._cgf_scalar, (0.1, 5, 1.1)),
        (NegativeBinomial._sample, (0, 0.5)),
        (NegativeBinomial._sample, (5, -0.5)),
    ],
)
def test_classmethods_reject_invalid_parameters(method, args):
    with pytest.raises(ValueError):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pmf_scalar", "negative_binomial_pmf_scalar", (3, 5, 0.6), 0.186),
    ("_cdf_scalar", "negative_binomial_cdf_scalar", (3, 5, 0.6), 0.663),
    ("_mean", "negative_binomial_mean", (5, 0.6), 3.333),
    ("_variance", "negative_binomial_variance", (5, 0.6), 5.556),
    ("_stddev", "negative_binomial_stddev", (5, 0.6), 2.357),
    ("_mgf_scalar", "negative_binomial_mgf_scalar", (0.1, 5, 0.6), 1.845),
    ("_cgf_scalar", "negative_binomial_cgf_scalar", (0.1, 5, 0.6), 0.612),
    ("_sample", "negative_binomial_sample", (5, 0.6), 4),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(NegativeBinomial, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    nb = NegativeBinomial(r=5, p=0.6)
    with pytest.raises(AttributeError):
        nb.extra = 123
