# tests/python_modules/test_normal.py
from unittest.mock import patch

import pytest

import fastdist.distributions.normal as normal_module
from fastdist.distributions.normal import Normal


@pytest.fixture
def mock_core():
    """
    Patch the internal C++ core for the duration of each test.
    """
    with patch.object(normal_module, "_core", autospec=True) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Constructor & representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    n = Normal(mu=0.0, sigma=1.0)
    assert n.mu == 0.0
    assert n.sigma == 1.0


def test_init_allows_none_parameters():
    n = Normal()
    assert n.mu is None
    assert n.sigma is None


@pytest.mark.parametrize("sigma", [0, -1, -10.5])
def test_init_invalid_sigma_raises(sigma):
    with pytest.raises(ValueError, match="sigma must be positive"):
        Normal(mu=0.0, sigma=sigma)


def test_repr():
    n = Normal(mu=1.5, sigma=2.5)
    assert repr(n) == "Normal(mu=1.5, sigma=2.5)"


# ---------------------------------------------------------------------------
# Validation behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0, -1])
def test_validate_params_rejects_non_positive_sigma(sigma):
    with pytest.raises(ValueError):
        Normal._validate_params(sigma=sigma)


def test_validate_params_accepts_positive_sigma():
    Normal._validate_params(sigma=1.0)  # should not raise


# ---------------------------------------------------------------------------
# Instance method delegation
# ---------------------------------------------------------------------------

def test_pdf_scalar_delegates_to_core(mock_core):
    mock_core.normal_pdf_scalar.return_value = 0.3989

    n = Normal(mu=0.0, sigma=1.0)
    result = n.pdf_scalar(0.0)

    mock_core.normal_pdf_scalar.assert_called_once_with(0.0, 0.0, 1.0)
    assert result == 0.3989


def test_logpdf_scalar_delegates_to_core(mock_core):
    mock_core.normal_logpdf_scalar.return_value = -0.9189

    n = Normal(mu=0.0, sigma=1.0)
    result = n.logpdf_scalar(0.0)

    mock_core.normal_logpdf_scalar.assert_called_once_with(0.0, 0.0, 1.0)
    assert result == -0.9189


def test_cdf_scalar_delegates_to_core(mock_core):
    mock_core.normal_cdf_scalar.return_value = 0.5

    n = Normal(mu=0.0, sigma=1.0)
    result = n.cdf_scalar(0.0)

    mock_core.normal_cdf_scalar.assert_called_once_with(0.0, 0.0, 1.0)
    assert result == 0.5


def test_z_score_delegates_to_core(mock_core):
    mock_core.z_score.return_value = 1.0

    n = Normal(mu=1.0, sigma=2.0)
    result = n.z_score(3.0)

    mock_core.z_score.assert_called_once_with(3.0, 1.0, 2.0)
    assert result == 1.0


def test_mgf_scalar_delegates_to_core(mock_core):
    mock_core.normal_mgf_scalar.return_value = 1.284

    n = Normal(mu=0.0, sigma=1.0)
    result = n.mgf_scalar(0.5)

    mock_core.normal_mgf_scalar.assert_called_once_with(0.5, 0.0, 1.0)
    assert result == 1.284


def test_cgf_scalar_delegates_to_core(mock_core):
    mock_core.normal_cgf_scalar.return_value = 0.25

    n = Normal(mu=0.0, sigma=1.0)
    result = n.cgf_scalar(0.5)

    mock_core.normal_cgf_scalar.assert_called_once_with(0.5, 0.0, 1.0)
    assert result == 0.25


def test_sample_delegates_to_core(mock_core):
    mock_core.normal_sample.return_value = 0.537

    n = Normal(mu=0.0, sigma=1.0)
    result = n.sample()

    mock_core.normal_sample.assert_called_once_with(0.0, 1.0)
    assert result == 0.537


def test_log_sample_delegates_to_core(mock_core):
    mock_core.normal_log_sample.return_value = -0.618

    n = Normal(mu=0.0, sigma=1.0)
    result = n.log_sample()

    mock_core.normal_log_sample.assert_called_once_with(0.0, 1.0)
    assert result == -0.618


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------

def test_mean_delegates_to_core(mock_core):
    mock_core.normal_mean.return_value = 5.0

    n = Normal(mu=5.0, sigma=1.0)
    result = n.mean()

    mock_core.normal_mean.assert_called_once_with(5.0)
    assert result == 5.0


def test_variance_delegates_to_core(mock_core):
    mock_core.normal_variance.return_value = 4.0

    n = Normal(mu=0.0, sigma=2.0)
    result = n.variance()

    mock_core.normal_variance.assert_called_once_with(2.0)
    assert result == 4.0


def test_stddev_delegates_to_core(mock_core):
    mock_core.normal_stddev.return_value = 2.0

    n = Normal(mu=0.0, sigma=2.0)
    result = n.stddev()

    mock_core.normal_stddev.assert_called_once_with(2.0)
    assert result == 2.0


# ---------------------------------------------------------------------------
# Batch methods
# ---------------------------------------------------------------------------
# BATCH AND CUDA TESTING NOT IMPLEMENTED YET
# def test_pdf_cpu_delegates_to_core(mock_core):
#     mock_core.normal_pdf_cpu.return_value = [0.3989, 0.2420]
#
#     n = Normal(mu=0.0, sigma=1.0)
#     result = n.pdf_cpu([0.0, 1.0])
#
#     mock_core.normal_pdf_cpu.assert_called_once_with([0.0, 1.0], 0.0, 1.0)
#     assert result == [0.3989, 0.2420]
#
#
# def test_pdf_cuda_delegates_to_core(mock_core):
#     mock_core.normal_pdf_cuda.return_value = [0.3989, 0.2420]
#
#     n = Normal(mu=0.0, sigma=1.0)
#     result = n.pdf_cuda([0.0, 1.0])
#
#     mock_core.normal_pdf_cuda.assert_called_once_with([0.0, 1.0], 0.0, 1.0)
#     assert result == [0.3989, 0.2420]


# ---------------------------------------------------------------------------
# Validation enforcement in classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, args",
    [
        (Normal._pdf_scalar, (0.0, 0.0, 0.0)),
        (Normal._logpdf_scalar, (0.0, 0.0, -1.0)),
        (Normal._cdf_scalar, (0.0, 0.0, 0.0)),
        (Normal._variance, (0.0,)),
        (Normal._stddev, (-1.0,)),
        (Normal._z_score, (0.0, 0.0, 0.0)),
        (Normal._mgf_scalar, (0.5, 0.0, 0.0)),
        (Normal._cgf_scalar, (0.5, 0.0, -1.0)),
        (Normal._sample, (0.0, 0.0)),
        (Normal._log_sample, (0.0, -1.0)),
        # (Normal._pdf_cpu, ([0.0], 0.0, 0.0)),
        # (Normal._pdf_cuda, ([0.0], 0.0, -1.0)),
    ],
)
def test_classmethods_reject_invalid_sigma(method, args):
    with pytest.raises(ValueError, match="sigma must be positive"):
        method(*args)


# ---------------------------------------------------------------------------
# Classmethod delegation with valid parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name, core_method_name, args, value", [
    ("_pdf_scalar", "normal_pdf_scalar", (0.0, 0.0, 1.0), 0.3989),
    ("_logpdf_scalar", "normal_logpdf_scalar", (0.0, 0.0, 1.0), -0.9189),
    ("_cdf_scalar", "normal_cdf_scalar", (0.0, 0.0, 1.0), 0.5),
    ("_mean", "normal_mean", (0.0,), 0.0),
    ("_variance", "normal_variance", (1.0,), 1.0),
    ("_stddev", "normal_stddev", (1.0,), 1.0),
    ("_mgf_scalar", "normal_mgf_scalar", (0.5, 0.0, 1.0), 1.284),
    ("_cgf_scalar", "normal_cgf_scalar", (0.5, 0.0, 1.0), 0.25),
    ("_sample", "normal_sample", (0.0, 1.0), 0.537),
    ("_log_sample", "normal_log_sample", (0.0, 1.0), -0.618),
    ("_z_score", "z_score", (1.0, 0.0, 1.0), 1.0),
])
def test_classmethods_delegate_to_core(mock_core, method_name, core_method_name, args, value):
    getattr(mock_core, core_method_name).return_value = value
    method = getattr(Normal, method_name)
    result = method(*args)
    getattr(mock_core, core_method_name).assert_called_once_with(*args)
    assert result == value


# ---------------------------------------------------------------------------
# Slots behavior
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    n = Normal(mu=0.0, sigma=1.0)
    with pytest.raises(AttributeError):
        n.foo = 123
