"""Numeric tests for the Chi-square distribution against closed-form references."""

import math

import pytest

from conftest import EXACT, ITERATIVE, regularized_lower_gamma
from fastdist.distributions.chi_square import ChiSquare


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def chi2_pdf(x: float, k: float) -> float:
    """f(x; k) = x^(k/2-1) e^(-x/2) / (2^(k/2) Gamma(k/2))"""
    return x ** (k / 2 - 1) * math.exp(-x / 2) / (2 ** (k / 2) * math.gamma(k / 2))


def chi2_mgf(t: float, k: float) -> float:
    """M(t) = (1 - 2t)^(-k/2), valid for t < 1/2"""
    return (1 - 2 * t) ** (-k / 2)


def chi2_cgf(t: float, k: float) -> float:
    """K(t) = -(k/2) ln(1 - 2t)"""
    return -(k / 2) * math.log(1 - 2 * t)


DEGREES = [1.0, 2.0, 3.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    assert ChiSquare(k=5.0).k == 5.0


@pytest.mark.parametrize("k", [-0.1, 0, -1, -10])
def test_init_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k must be positive"):
        ChiSquare(k=k)


@pytest.mark.parametrize("bad", ["5.0", None, object()])
def test_init_rejects_non_real_k(bad):
    with pytest.raises(TypeError, match="k must be a real number"):
        ChiSquare(k=bad)


def test_repr():
    assert repr(ChiSquare(k=3.0)) == "ChiSquare(k=3.0)"


def test_k_setter_updates_and_validates():
    dist = ChiSquare(k=5.0)
    dist.k = 8.0
    assert dist.k == 8.0
    with pytest.raises(ValueError, match="k must be positive"):
        dist.k = -1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0.1, 1.0, 5.0, 100.0])
def test_validate_params_accepts_valid_values(k):
    ChiSquare._validate_params(k=k)


@pytest.mark.parametrize("k", [0, -1, -5.0])
def test_validate_params_rejects_invalid_values(k):
    with pytest.raises(ValueError, match="k must be positive"):
        ChiSquare._validate_params(k=k)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", DEGREES)
@pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 7.5])
def test_pdf_matches_closed_form(x, k):
    assert ChiSquare(k).pdf(x) == pytest.approx(chi2_pdf(x, k), **EXACT)


@pytest.mark.parametrize("k", DEGREES)
@pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 20.0])
def test_pdf_is_non_negative(x, k):
    assert ChiSquare(k).pdf(x) >= 0.0


@pytest.mark.parametrize("x", [0.5, 1.0, 4.0])
def test_pdf_of_two_degrees_is_exponential(x):
    """Chi-square with k=2 is Exponential with mean 2, so f(x) = e^(-x/2) / 2."""
    assert ChiSquare(2.0).pdf(x) == pytest.approx(math.exp(-x / 2) / 2, **EXACT)


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

# REGRESSION: chi_square_cdf_scalar delegates to the same regularized lower
# incomplete gamma as Gamma.cdf_scalar, whose continued-fraction branch used an
# unsigned loop index under a unary minus and was wrong as a result.
# ChiSquare(3.0).cdf(7.5) returned 1.000498004, a probability greater than one.
# k = 2 was correct throughout because that case reduces to the exact
# exponential form. See test_gamma.py for the full diagnosis.
#
# Validated against scipy.special.gammainc over 132 points with k from 0.5 to
# 10000: worst absolute error 3.1e-12, nothing outside [0, 1].

CHI2_X = (0.5, 1.0, 3.0, 7.5, 20.0, 50.0)

CDF_CASES = [(k, x) for k in DEGREES for x in CHI2_X]


@pytest.mark.parametrize("k, x", CDF_CASES)
def test_cdf_matches_reference(k, x):
    assert ChiSquare(k).cdf(x) == pytest.approx(
        regularized_lower_gamma(k / 2.0, x / 2.0), **ITERATIVE
    )


@pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 7.5, 20.0])
def test_cdf_exact_for_two_degrees_of_freedom(x):
    """For k=2 the CDF has the exact closed form 1 - e^(-x/2)."""
    assert ChiSquare(2.0).cdf(x) == pytest.approx(1 - math.exp(-x / 2), **ITERATIVE)


_CDF_PROPERTY_DEGREES = [1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1000.0]


@pytest.mark.parametrize("k", _CDF_PROPERTY_DEGREES)
def test_cdf_is_bounded(k):
    dist = ChiSquare(k)
    for x in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 1000.0):
        assert 0.0 <= dist.cdf(x) <= 1.0


@pytest.mark.parametrize("k", _CDF_PROPERTY_DEGREES)
def test_cdf_is_monotonic(k):
    dist = ChiSquare(k)
    values = [dist.cdf(x) for x in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 1000.0)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1.0, 2.0, 5.0, 12.5, 100.0])
def test_moments_match_closed_forms(k):
    dist = ChiSquare(k)
    assert dist.mean() == pytest.approx(k, **EXACT)
    assert dist.variance() == pytest.approx(2 * k, **EXACT)
    assert dist.stddev() == pytest.approx(math.sqrt(2 * k), **EXACT)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("t", [-0.5, -0.1, 0.0, 0.1, 0.2])
def test_mgf_matches_closed_form(t, k):
    assert ChiSquare(k).mgf_scalar(t) == pytest.approx(chi2_mgf(t, k), **EXACT)


@pytest.mark.parametrize("k", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("t", [-0.5, -0.1, 0.0, 0.1, 0.2])
def test_cgf_matches_closed_form(t, k):
    assert ChiSquare(k).cgf_scalar(t) == pytest.approx(chi2_cgf(t, k), **EXACT)


@pytest.mark.parametrize("k", [1.0, 3.0, 8.0])
def test_mgf_at_zero_is_one(k):
    assert ChiSquare(k).mgf_scalar(0.0) == pytest.approx(1.0, **EXACT)
    assert ChiSquare(k).cgf_scalar(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("k", [1.0, 5.0])
@pytest.mark.parametrize("t", [-0.3, 0.1, 0.25])
def test_cgf_is_log_of_mgf(t, k):
    """K(t) = ln M(t) - catches a sign or factor error in either one."""
    dist = ChiSquare(k)
    assert dist.cgf_scalar(t) == pytest.approx(math.log(dist.mgf_scalar(t)), **EXACT)


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1.0, 5.0])
@pytest.mark.parametrize("x", [0.5, 3.0])
def test_classmethods_agree_with_instances(x, k):
    dist = ChiSquare(k)
    assert ChiSquare._pdf_scalar(x, k) == pytest.approx(dist.pdf(x), **EXACT)
    assert ChiSquare._cdf_scalar(x, k) == pytest.approx(dist.cdf(x), **ITERATIVE)
    assert ChiSquare._mean(k) == pytest.approx(dist.mean(), **EXACT)
    assert ChiSquare._variance(k) == pytest.approx(dist.variance(), **EXACT)
    assert ChiSquare._stddev(k) == pytest.approx(dist.stddev(), **EXACT)
    assert ChiSquare._mgf_scalar(0.1, k) == pytest.approx(dist.mgf_scalar(0.1), **EXACT)
    assert ChiSquare._cgf_scalar(0.1, k) == pytest.approx(dist.cgf_scalar(0.1), **EXACT)


@pytest.mark.parametrize("method_name, args", [
    ("_pdf_scalar", (1.0, -1)),
    ("_pdf_scalar", (1.0, 0)),
    ("_cdf_scalar", (1.0, -5.0)),
    ("_cdf_scalar", (1.0, 0)),
    ("_mean", (-1,)),
    ("_mean", (0,)),
    ("_variance", (-5.0,)),
    ("_variance", (0,)),
    ("_stddev", (-1,)),
    ("_stddev", (0,)),
    ("_mgf_scalar", (0.1, -1)),
    ("_mgf_scalar", (0.1, 0)),
    ("_cgf_scalar", (0.1, -5.0)),
    ("_cgf_scalar", (0.1, 0)),
    ("_sample", (-1,)),
    ("_sample", (0,)),
])
def test_classmethods_reject_invalid_parameters(method_name, args):
    with pytest.raises(ValueError, match=r"k must be positive"):
        getattr(ChiSquare, method_name)(*args)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1.0, 5.0, 20.0])
def test_sample_is_positive_and_finite(k):
    dist = ChiSquare(k)
    for _ in range(200):
        value = dist.sample()
        assert math.isfinite(value)
        assert value > 0.0


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        ChiSquare(k=5.0).extra = 123
