# tests/python_modules/test_geometric.py
from unittest.mock import patch

import pytest

import fastdist.distributions.geometric as geometric_module
from fastdist.distributions.geometric import Geometric


@pytest.fixture
def mock_core():
    """Patch the internal C++ core for the duration of each test."""
    with patch.object(geometric_module, "_core", create=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    g = Geometric(p=0.5)
    assert g.p == 0.5


@pytest.mark.parametrize("p", [0, -0.1, -1, 1.1, 2])
def test_init_invalid_p_raises(p):
    with pytest.raises(ValueError, match=r"p must be in the interval \(0, 1\]"):
        Geometric(p=p)


def test_repr():
    g = Geometric(p=0.3)
    assert repr(g) == "Geometric(p=0.3)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.001, 0.5, 1.0])
def test_validate_params_accepts_valid_values(p):
    Geometric._validate_params(p=p)


@pytest.mark.parametrize("p", [0, -0.5, 1.1, 2])
def test_validate_params_rejects_invalid_values(p):
    with pytest.raises(ValueError, match=r"p must be in the interval \(0, 1\]"):
        Geometric._validate_params(p=p)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pmf_scalar_delegates_to_core(mock_core):
    mock_core.geometric_pmf_scalar.return_value = 0.125

    g = Geometric(p=0.5)
    result = g.pmf_scalar(3)

    mock_core.geometric_pmf_scalar.assert_called_once_with(3, 0.5)
    assert result == 0.125


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.geometric_cdf_scalar.return_value = 0.875

    g = Geometric(p=0.5)
    result = g.cdf_scalar(3)

    mock_core.geometric_cdf_scalar.assert_called_once_with(3, 0.5)
    assert result == 0.875


def test_mean_delegates_to_core(mock_core):
    mock_core.geometric_mean.return_value = 2.0

    g = Geometric(p=0.5)
    result = g.mean()

    mock_core.geometric_mean.assert_called_once_with(0.5)
    assert result == 2.0


def test_variance_delegates_to_core(mock_core):
    mock_core.geometric_variance.return_value = 2.0

    g = Geometric(p=0.5)
    result = g.variance()

    mock_core.geometric_variance.assert_called_once_with(0.5)
    assert result == 2.0


def test_stddev_delegates_to_core(mock_core):
    mock_core.geometric_stddev.return_value = 1.414

    g = Geometric(p=0.5)
    result = g.stddev()

    mock_core.geometric_stddev.assert_called_once_with(0.5)
    assert result == 1.414


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.geometric_mgf_scalar.return_value = 1.648

    g = Geometric(p=0.5)
    result = g.mgf_scalar(0.2)

    mock_core.geometric_mgf_scalar.assert_called_once_with(0.2, 0.5)
    assert result == 1.648


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.geometric_cgf_scalar.return_value = 0.499

    g = Geometric(p=0.5)
    result = g.cgf_scalar(0.2)

    mock_core.geometric_cgf_scalar.assert_called_once_with(0.2, 0.5)
    assert result == 0.499


def test_sample_delegates_to_core(mock_core):
    mock_core.geometric_sample.return_value = 3

    g = Geometric(p=0.5)
    result = g.sample()

    mock_core.geometric_sample.assert_called_once_with(0.5)
    assert result == 3


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (Geometric._pmf_scalar, (3, 0)),
        (Geometric._pmf_scalar, (3, -0.1)),
        (Geometric._pmf_scalar, (3, 1.1)),
        (Geometric._cdf_scalar, (3, 0)),
        (Geometric._cdf_scalar, (3, -0.5)),
        (Geometric._cdf_scalar, (3, 2)),
        (Geometric._mean, (0,)),
        (Geometric._mean, (-0.1,)),
        (Geometric._mean, (1.5,)),
        (Geometric._variance, (0,)),
        (Geometric._variance, (-1,)),
        (Geometric._stddev, (0,)),
        (Geometric._stddev, (1.1,)),
        (Geometric._mgf_scalar, (0.2, 0)),
        (Geometric._mgf_scalar, (0.2, -0.1)),
        (Geometric._cgf_scalar, (0.2, 0)),
        (Geometric._cgf_scalar, (0.2, 1.5)),
        (Geometric._sample, (0,)),
        (Geometric._sample, (-0.5,)),
    ],
)
def test_classmethods_reject_invalid_parameters(method, args):
    with pytest.raises(ValueError, match=r"p must be in the interval \(0, 1\]"):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pmf_scalar", "geometric_pmf_scalar", (3, 0.5), 0.125),
    ("_cdf_scalar", "geometric_cdf_scalar", (3, 0.5), 0.875),
    ("_mean", "geometric_mean", (0.5,), 2.0),
    ("_variance", "geometric_variance", (0.5,), 2.0),
    ("_stddev", "geometric_stddev", (0.5,), 1.414),
    ("_mgf_scalar", "geometric_mgf_scalar", (0.2, 0.5), 1.648),
    ("_cgf_scalar", "geometric_cgf_scalar", (0.2, 0.5), 0.499),
    ("_sample", "geometric_sample", (0.5,), 3),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(Geometric, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    g = Geometric(p=0.5)
    with pytest.raises(AttributeError):
        g.extra = 123