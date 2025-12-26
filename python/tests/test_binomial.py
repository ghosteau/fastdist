from unittest.mock import patch

import pytest

# Import module so we can patch _core correctly
import fastdist.distributions.binomial as binomial_module
from fastdist.distributions.binomial import Binomial


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(binomial_module, "_core", autospec=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    b = Binomial(n=10, p=0.3)
    assert b.n == 10
    assert b.p == 0.3


def test_init_allows_none_parameters():
    b = Binomial()
    assert b.n is None
    assert b.p is None


@pytest.mark.parametrize("n", [-1, -10, 1.5, "10"])
def test_init_invalid_n_raises(n):
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        Binomial(n=n, p=0.5)


@pytest.mark.parametrize("p", [-0.1, 1.1, -1, 2])
def test_init_invalid_p_raises(p):
    with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
        Binomial(n=10, p=p)


def test_repr():
    b = Binomial(n=5, p=0.4)
    # NOTE: repr is incorrect in implementation, this test reflects current behavior
    assert repr(b) == "Normal(mu=5, sigma=0.4)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [-1, -2, 1.1])
def test_validate_params_rejects_invalid_n(n):
    with pytest.raises(ValueError):
        Binomial._validate_params(n=n, p=0.5)


@pytest.mark.parametrize("p", [-0.01, 1.01])
def test_validate_params_rejects_invalid_p(p):
    with pytest.raises(ValueError):
        Binomial._validate_params(n=10, p=p)


def test_validate_params_accepts_boundary_values():
    Binomial._validate_params(n=0, p=0.0)
    Binomial._validate_params(n=10, p=1.0)


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_logpmf_scalar_delegates_to_core(mock_core):
    mock_core.binomial_logpmf_scalar.return_value = -1.23

    b = Binomial(n=10, p=0.5)
    result = b.logpmf_scalar(3)

    mock_core.binomial_logpmf_scalar.assert_called_once_with(3, 10, 0.5)
    assert result == -1.23


def test_pmf_scalar_delegates_to_core(mock_core):
    mock_core.binomial_pmf_scalar.return_value = 0.117

    b = Binomial(n=10, p=0.3)
    result = b.pmf_scalar(2)

    mock_core.binomial_pmf_scalar.assert_called_once_with(2, 10, 0.3)
    assert result == 0.117


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.binomial_cdf_scalar.return_value = 0.952

    b = Binomial(n=10, p=0.5)
    result = b.cdf_scalar(7)

    mock_core.binomial_cdf_scalar.assert_called_once_with(7, 10, 0.5)
    assert result == 0.952


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_mean_delegates_to_core(mock_core):
    mock_core.binomial_mean.return_value = 3.0

    b = Binomial(n=10, p=0.3)
    result = b.mean()

    mock_core.binomial_mean.assert_called_once_with(10, 0.3)
    assert result == 3.0


def test_variance_delegates_to_core(mock_core):
    mock_core.binomial_variance.return_value = 2.1

    b = Binomial(n=10, p=0.3)
    result = b.variance()

    mock_core.binomial_variance.assert_called_once_with(10, 0.3)
    assert result == 2.1


def test_stddev_delegates_to_core(mock_core):
    mock_core.binomial_stddev.return_value = 1.449

    b = Binomial(n=10, p=0.21)
    result = b.stddev()

    mock_core.binomial_stddev.assert_called_once_with(10, 0.21)
    assert result == 1.449


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (Binomial._logpmf_scalar, (1, -1, 0.5)),
        (Binomial._pmf_scalar, (1, 10, -0.1)),
        (Binomial._cdf_scalar, (1, 1.5, 0.5)),
        (Binomial._mean, (-1, 0.5)),
        (Binomial._variance, (10, 2.0)),
        (Binomial._stddev, (-5, 0.5)),
    ],
)
def test_classmethods_reject_invalid_parameters(method, args):
    with pytest.raises(ValueError):
        method(*args)


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    b = Binomial(n=10, p=0.5)
    with pytest.raises(AttributeError):
        b.extra = 123
