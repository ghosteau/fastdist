# tests/test_normal.py
import math

import numpy as np
import pytest

from fastdist.distributions.normal import Normal


# ---------------------------
# Fixtures
# ---------------------------
@pytest.fixture
def standard_normal():
    return Normal(mu=0, sigma=1)


@pytest.fixture
def shifted_normal():
    return Normal(mu=5, sigma=2)


# ---------------------------
# Constructor & Properties
# ---------------------------
def test_constructor_valid(standard_normal):
    assert standard_normal.mu == 0.0
    assert standard_normal.sigma == 1.0


def test_constructor_invalid_mu():
    with pytest.raises(TypeError):
        Normal(mu="invalid", sigma=1)
    with pytest.raises(ValueError):
        Normal(mu=float('inf'), sigma=1)


def test_constructor_invalid_sigma():
    with pytest.raises(TypeError):
        Normal(mu=0, sigma="invalid")
    with pytest.raises(ValueError):
        Normal(mu=0, sigma=0)
    with pytest.raises(ValueError):
        Normal(mu=0, sigma=-1)
    with pytest.raises(ValueError):
        Normal(mu=0, sigma=float('nan'))


def test_setters(standard_normal):
    standard_normal.mu = 2
    assert standard_normal.mu == 2.0
    standard_normal.sigma = 3
    assert standard_normal.sigma == 3.0
    with pytest.raises(ValueError):
        standard_normal.sigma = 0


# ---------------------------
# PDF / logPDF / CDF
# ---------------------------
def test_pdf_scalar(standard_normal):
    x = 0
    pdf_val = standard_normal.pdf(x)
    expected = 1 / math.sqrt(2 * math.pi)
    assert math.isclose(pdf_val, expected, rel_tol=1e-9)


def test_logpdf_scalar(standard_normal):
    x = 0
    logpdf_val = standard_normal.logpdf(x)
    expected = math.log(1 / math.sqrt(2 * math.pi))
    assert math.isclose(logpdf_val, expected, rel_tol=1e-9)


def test_cdf_scalar(standard_normal):
    x = 0
    cdf_val = standard_normal.cdf(x)
    assert math.isclose(cdf_val, 0.5, rel_tol=1e-9)


def test_pdf_vector(standard_normal):
    x = [-1, 0, 1]
    result = standard_normal.pdf(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert math.isclose(result[1], 1 / math.sqrt(2 * math.pi), rel_tol=1e-9)


# ---------------------------
# MGF / CGF
# ---------------------------
def test_mgf_cgf_scalar(shifted_normal):
    t = 1.0
    mgf_val = shifted_normal.mgf(t)
    expected_mgf = math.exp(shifted_normal.mu * t + 0.5 * shifted_normal.sigma ** 2 * t ** 2)
    assert math.isclose(mgf_val, expected_mgf, rel_tol=1e-9)

    cgf_val = shifted_normal.cgf(t)
    expected_cgf = math.log(expected_mgf)
    assert math.isclose(cgf_val, expected_cgf, rel_tol=1e-9)


# ---------------------------
# Sampling
# ---------------------------
def test_sample(standard_normal):
    val = standard_normal.sample()
    assert isinstance(val, float)


def test_log_sample(standard_normal):
    val = standard_normal.log_sample()
    assert isinstance(val, float)


# ---------------------------
# Z-score
# ---------------------------
def test_z_score_scalar(standard_normal):
    z = standard_normal.z_score(1)
    assert math.isclose(z, 1.0)


# ---------------------------
# CUDA availability
# ---------------------------
def test_cuda_availability():
    available = Normal.is_cuda_available()
    assert isinstance(available, bool)


# ---------------------------
# Scalar class methods
# ---------------------------
def test_pdf_scalar_class_method():
    val = Normal._pdf_scalar(0, mu=0, sigma=1)
    expected = 1 / math.sqrt(2 * math.pi)
    assert math.isclose(val, expected, rel_tol=1e-9)


def test_logpdf_scalar_class_method():
    val = Normal._logpdf_scalar(0, mu=0, sigma=1)
    expected = math.log(1 / math.sqrt(2 * math.pi))
    assert math.isclose(val, expected, rel_tol=1e-9)


def test_cdf_scalar_class_method():
    val = Normal._cdf_scalar(0, mu=0, sigma=1)
    assert math.isclose(val, 0.5, rel_tol=1e-9)


def test_logpdf_accepts_an_array(standard_normal):
    """
    The vectorised path once dropped step_size when forwarding to the backend,
    which raised TypeError for every array input. Only the scalar path was
    covered, so nothing caught it.
    """
    x = np.array([-1.0, 0.0, 1.0])
    got = standard_normal.logpdf(x)
    expected = [math.log(1.0 / math.sqrt(2.0 * math.pi)) - v * v / 2.0 for v in x]
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Lower-tail precision
#
# 0.5 * (1 + erf(z)) cancels to nothing far below the mean, so the CDF lost
# all relative precision below about -8 sigma and returned exactly 0 at -10.
# Tail probabilities are what p-values are made of, so they are checked here
# against erfc directly, on both the scalar and the batch path.
# ---------------------------------------------------------------------------

def _phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


@pytest.mark.parametrize("x", [-5.0, -8.0, -10.0, -20.0, -37.0])
def test_cdf_keeps_relative_precision_in_the_lower_tail(x):
    import fastdist._fastdist as core
    expected = _phi(x)
    assert core.normal_cdf_scalar(x, 0.0, 1.0) == pytest.approx(expected, rel=1e-12)
    batch = core.normal_cdf_cpu(np.array([x]), 0.0, 1.0, 0.0)
    assert batch[0] == pytest.approx(expected, rel=1e-12)
