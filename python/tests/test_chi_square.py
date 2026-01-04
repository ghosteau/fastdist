# tests/python_modules/test_chi_square.py
from unittest.mock import patch

import pytest

import fastdist.distributions.chi_square as chi_square_module
from fastdist.distributions.chi_square import ChiSquare


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
    with patch.object(chi_square_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    cs = ChiSquare(k=5.0)
    assert cs.k == 5.0


@pytest.mark.parametrize("k", [-0.1, 0, -1, -10])
def test_init_invalid_k_raises(k):
    with pytest.raises(ValueError, match="k must be positive"):
        ChiSquare(k=k)


def test_repr():
    cs = ChiSquare(k=3.0)
    assert repr(cs) == "ChiSquare(k=3.0)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0.1, 1.0, 5.0, 100.0])
def test_validate_params_accepts_valid_values(k):
    ChiSquare._validate_params(k=k)


@pytest.mark.parametrize("k", [0, -1, -5.0])
def test_validate_params_rejects_invalid_values(k):
    with pytest.raises(ValueError, match="k must be positive"):
        ChiSquare._validate_params(k=k)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pdf_delegates_to_core(mock_core):
    mock_core.chi_square_pdf_scalar.return_value = 0.154

    cs = ChiSquare(k=5.0)
    result = cs.pdf(3.0)

    mock_core.chi_square_pdf_scalar.assert_called_once_with(3.0, 5.0)
    assert result == 0.154


def test_cdf_delegates_to_core(mock_core):
    mock_core.chi_square_cdf_scalar.return_value = 0.416

    cs = ChiSquare(k=5.0)
    result = cs.cdf(3.0)

    mock_core.chi_square_cdf_scalar.assert_called_once_with(3.0, 5.0)
    assert result == 0.416


def test_mean_delegates_to_core(mock_core):
    mock_core.chi_square_mean.return_value = 5.0

    cs = ChiSquare(k=5.0)
    result = cs.mean()

    mock_core.chi_square_mean.assert_called_once_with(5.0)
    assert result == 5.0


def test_variance_delegates_to_core(mock_core):
    mock_core.chi_square_variance.return_value = 10.0

    cs = ChiSquare(k=5.0)
    result = cs.variance()

    mock_core.chi_square_variance.assert_called_once_with(5.0)
    assert result == 10.0


def test_stddev_delegates_to_core(mock_core):
    mock_core.chi_square_stddev.return_value = 3.162

    cs = ChiSquare(k=5.0)
    result = cs.stddev()

    mock_core.chi_square_stddev.assert_called_once_with(5.0)
    assert result == 3.162


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.chi_square_mgf_scalar.return_value = 1.789

    cs = ChiSquare(k=5.0)
    result = cs.mgf_scalar(0.1)

    mock_core.chi_square_mgf_scalar.assert_called_once_with(0.1, 5.0)
    assert result == 1.789


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.chi_square_cgf_scalar.return_value = 0.581

    cs = ChiSquare(k=5.0)
    result = cs.cgf_scalar(0.1)

    mock_core.chi_square_cgf_scalar.assert_called_once_with(0.1, 5.0)
    assert result == 0.581


def test_sample_delegates_to_core(mock_core):
    mock_core.chi_square_sample.return_value = 4.823

    cs = ChiSquare(k=5.0)
    result = cs.sample()

    mock_core.chi_square_sample.assert_called_once_with(5.0)
    assert result == 4.823


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (ChiSquare._pdf_scalar, (1.0, -1)),
        (ChiSquare._pdf_scalar, (1.0, 0)),
        (ChiSquare._cdf_scalar, (1.0, -5.0)),
        (ChiSquare._cdf_scalar, (1.0, 0)),
        (ChiSquare._mean, (-1,)),
        (ChiSquare._mean, (0,)),
        (ChiSquare._variance, (-5.0,)),
        (ChiSquare._variance, (0,)),
        (ChiSquare._stddev, (-1,)),
        (ChiSquare._stddev, (0,)),
        (ChiSquare._mgf_scalar, (0.1, -1)),
        (ChiSquare._mgf_scalar, (0.1, 0)),
        (ChiSquare._cgf_scalar, (0.1, -5.0)),
        (ChiSquare._cgf_scalar, (0.1, 0)),
        (ChiSquare._sample, (-1,)),
        (ChiSquare._sample, (0,)),
    ],
)
def test_classmethods_reject_invalid_parameters(method, args):
    with pytest.raises(ValueError, match=r"k must be positive"):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pdf_scalar", "chi_square_pdf_scalar", (3.0, 5.0), 0.154),
    ("_cdf_scalar", "chi_square_cdf_scalar", (3.0, 5.0), 0.416),
    ("_mean", "chi_square_mean", (5.0,), 5.0),
    ("_variance", "chi_square_variance", (5.0,), 10.0),
    ("_stddev", "chi_square_stddev", (5.0,), 3.162),
    ("_mgf_scalar", "chi_square_mgf_scalar", (0.1, 5.0), 1.789),
    ("_cgf_scalar", "chi_square_cgf_scalar", (0.1, 5.0), 0.581),
    ("_sample", "chi_square_sample", (5.0,), 4.823),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(ChiSquare, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    cs = ChiSquare(k=5.0)
    with pytest.raises(AttributeError):
        cs.extra = 123
