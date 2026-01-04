# tests/python_modules/test_gamma.py
from unittest.mock import patch

import pytest

import fastdist.distributions.gamma as gamma_module
from fastdist.distributions.gamma import Gamma


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
    with patch.object(gamma_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    g = Gamma(alpha=2.0, theta=3.0)
    assert g.alpha == 2.0
    assert g.theta == 3.0


def test_init_allows_none_parameters():
    g = Gamma()
    assert g.alpha is None
    assert g.theta is None


@pytest.mark.parametrize("alpha,theta", [
    (-0.1, 1.0),
    (0, 1.0),
    (-1, 1.0),
    (1.0, -0.1),
    (1.0, 0),
    (1.0, -1),
    (-1, -1),
])
def test_init_invalid_parameters_raises(alpha, theta):
    with pytest.raises(ValueError, match="must be positive"):
        Gamma(alpha=alpha, theta=theta)


def test_repr():
    g = Gamma(alpha=2.0, theta=3.0)
    assert repr(g) == "Gamma(alpha=2.0, theta=3.0)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha,theta", [
    (0.1, 0.1),
    (1.0, 1.0),
    (2.5, 3.5),
    (100, 200),
])
def test_validate_params_accepts_valid_values(alpha, theta):
    Gamma._validate_params(alpha=alpha, theta=theta)


@pytest.mark.parametrize("alpha,theta", [
    (0, 1.0),
    (-1, 1.0),
    (1.0, 0),
    (1.0, -1),
])
def test_validate_params_rejects_invalid_values(alpha, theta):
    with pytest.raises(ValueError, match="must be positive"):
        Gamma._validate_params(alpha=alpha, theta=theta)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pmf_scalar_delegates_to_core(mock_core):
    mock_core.gamma_pdf_scalar.return_value = 0.234

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.pmf_scalar(5.0)

    mock_core.gamma_pdf_scalar.assert_called_once_with(5.0, 2.0, 3.0)
    assert result == 0.234


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.gamma_cdf_scalar.return_value = 0.456

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.cdf_scalar(5.0)

    mock_core.gamma_cdf_scalar.assert_called_once_with(5.0, 2.0, 3.0)
    assert result == 0.456


def test_mean_delegates_to_core(mock_core):
    mock_core.gamma_mean.return_value = 6.0

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.mean()

    mock_core.gamma_mean.assert_called_once_with(2.0, 3.0)
    assert result == 6.0


def test_variance_delegates_to_core(mock_core):
    mock_core.gamma_variance.return_value = 18.0

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.variance()

    mock_core.gamma_variance.assert_called_once_with(2.0, 3.0)
    assert result == 18.0


def test_stddev_delegates_to_core(mock_core):
    mock_core.gamma_stddev.return_value = 4.243

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.stddev()

    mock_core.gamma_stddev.assert_called_once_with(2.0, 3.0)
    assert result == 4.243


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.gamma_mgf_scalar.return_value = 1.789

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.mgf_scalar(0.1)

    mock_core.gamma_mgf_scalar.assert_called_once_with(0.1, 2.0, 3.0)
    assert result == 1.789


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.gamma_cgf_scalar.return_value = 0.581

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.cgf_scalar(0.1)

    mock_core.gamma_cgf_scalar.assert_called_once_with(0.1, 2.0, 3.0)
    assert result == 0.581


def test_sample_delegates_to_core(mock_core):
    mock_core.gamma_sample.return_value = 5.823

    g = Gamma(alpha=2.0, theta=3.0)
    result = g.sample()

    mock_core.gamma_sample.assert_called_once_with(2.0, 3.0)
    assert result == 5.823


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (Gamma._pdf_scalar, (1.0, -1, 1.0)),
        (Gamma._pdf_scalar, (1.0, 0, 1.0)),
        (Gamma._pdf_scalar, (1.0, 1.0, -1)),
        (Gamma._pdf_scalar, (1.0, 1.0, 0)),
        (Gamma._cdf_scalar, (1.0, -5.0, 1.0)),
        (Gamma._cdf_scalar, (1.0, 1.0, -5.0)),
        (Gamma._mean, (-1, 1.0)),
        (Gamma._mean, (1.0, -1)),
        (Gamma._variance, (0, 1.0)),
        (Gamma._variance, (1.0, 0)),
        (Gamma._stddev, (-1, 1.0)),
        (Gamma._stddev, (1.0, -1)),
        (Gamma._mgf_scalar, (0.1, -1, 1.0)),
        (Gamma._mgf_scalar, (0.1, 1.0, -1)),
        (Gamma._cgf_scalar, (0.1, -5.0, 1.0)),
        (Gamma._cgf_scalar, (0.1, 1.0, -5.0)),
        (Gamma._sample, (-1, 1.0)),
        (Gamma._sample, (1.0, -1)),
    ],
)
def test_classmethods_reject_invalid_parameters(method, args):
    with pytest.raises(ValueError, match=r"must be positive"):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pdf_scalar", "gamma_pdf_scalar", (5.0, 2.0, 3.0), 0.234),
    ("_cdf_scalar", "gamma_cdf_scalar", (5.0, 2.0, 3.0), 0.456),
    ("_mean", "gamma_mean", (2.0, 3.0), 6.0),
    ("_variance", "gamma_variance", (2.0, 3.0), 18.0),
    ("_stddev", "gamma_stddev", (2.0, 3.0), 4.243),
    ("_mgf_scalar", "gamma_mgf_scalar", (0.1, 2.0, 3.0), 1.789),
    ("_cgf_scalar", "gamma_cgf_scalar", (0.1, 2.0, 3.0), 0.581),
    ("_sample", "gamma_sample", (2.0, 3.0), 5.823),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(Gamma, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    g = Gamma(alpha=2.0, theta=3.0)
    with pytest.raises(AttributeError):
        g.extra = 123
