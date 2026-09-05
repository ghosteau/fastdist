"""
Numeric tests for the Gamma distribution against closed-form references.

`theta` is the *scale* parameter: mean = alpha * theta, variance = alpha * theta^2.
"""

import math

import pytest

from conftest import EXACT, ITERATIVE, regularized_lower_gamma
from fastdist.distributions.gamma import Gamma


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def gamma_pdf(x: float, alpha: float, theta: float) -> float:
    """f(x; a, th) = x^(a-1) e^(-x/th) / (th^a Gamma(a))"""
    return x ** (alpha - 1) * math.exp(-x / theta) / (theta ** alpha * math.gamma(alpha))


def gamma_mgf(t: float, alpha: float, theta: float) -> float:
    """M(t) = (1 - th t)^(-a), valid for t < 1/th"""
    return (1 - theta * t) ** (-alpha)


def gamma_cgf(t: float, alpha: float, theta: float) -> float:
    """K(t) = -a ln(1 - th t)"""
    return -alpha * math.log(1 - theta * t)


PARAMS = [(1.0, 1.0), (2.0, 3.0), (5.0, 2.0), (0.5, 4.0), (3.0, 0.5)]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    dist = Gamma(alpha=2.0, theta=3.0)
    assert dist.alpha == 2.0
    assert dist.theta == 3.0


@pytest.mark.parametrize("alpha, theta", [(0, 1.0), (-1, 1.0), (1.0, 0), (1.0, -2.0)])
def test_init_rejects_non_positive_parameters(alpha, theta):
    with pytest.raises(ValueError, match="must be positive"):
        Gamma(alpha=alpha, theta=theta)


@pytest.mark.parametrize("bad", ["2.0", object()])
def test_init_rejects_non_real_parameters(bad):
    with pytest.raises(TypeError, match="must be a real number"):
        Gamma(alpha=bad, theta=1.0)


def test_init_rejects_none_parameter():
    """
    None is rejected, but only incidentally: _validate_params skips its checks
    when a parameter is None, so the failure surfaces later as
    "float() argument must be a string or a real number" rather than the
    intended "alpha must be a real number".
    """
    with pytest.raises(TypeError):
        Gamma(alpha=None, theta=1.0)


def test_repr():
    assert repr(Gamma(alpha=2.0, theta=3.0)) == "Gamma(alpha=2.0, theta=3.0)"


def test_property_setters_update_and_validate():
    dist = Gamma(alpha=2.0, theta=3.0)
    dist.alpha = 4.0
    dist.theta = 5.0
    assert dist.alpha == 4.0
    assert dist.theta == 5.0
    with pytest.raises(ValueError, match="alpha must be positive"):
        dist.alpha = 0
    with pytest.raises(ValueError, match="theta must be positive"):
        dist.theta = -1.0


# ---------------------------------------------------------------------------
# PDF
#
# NOTE: the instance method is named `pmf_scalar` even though Gamma is a
# continuous distribution and the corresponding classmethod is `_pdf_scalar`.
# The tests follow the shipped API; the naming inconsistency is tracked
# separately.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, theta", PARAMS)
@pytest.mark.parametrize("x", [0.5, 1.0, 3.0, 8.0])
def test_pdf_matches_closed_form(x, alpha, theta):
    assert Gamma(alpha, theta).pmf_scalar(x) == pytest.approx(
        gamma_pdf(x, alpha, theta), **EXACT
    )


@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("x", [0.25, 1.0, 4.0])
def test_pdf_with_unit_shape_is_exponential(x, theta):
    """Gamma(1, th) is Exponential with rate 1/th."""
    assert Gamma(1.0, theta).pmf_scalar(x) == pytest.approx(
        math.exp(-x / theta) / theta, **EXACT
    )


@pytest.mark.parametrize("alpha, theta", PARAMS)
@pytest.mark.parametrize("x", [0.1, 2.0, 15.0])
def test_pdf_is_non_negative(x, alpha, theta):
    assert Gamma(alpha, theta).pmf_scalar(x) >= 0.0


@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("x", [0.25, 1.0, 4.0])
def test_pdf_with_unit_shape_is_exponential(x, theta):
    """Gamma(1, th) is Exponential with rate 1/th."""
    assert Gamma(1.0, theta).pmf_scalar(x) == pytest.approx(
        math.exp(-x / theta) / theta, **EXACT
    )


@pytest.mark.parametrize("alpha, theta", PARAMS)
@pytest.mark.parametrize("x", [0.1, 2.0, 15.0])
def test_pdf_is_non_negative(x, alpha, theta):
    assert Gamma(alpha, theta).pmf_scalar(x) >= 0.0


# ---------------------------------------------------------------------------
# CDF
#
# REGRESSION: the regularized lower incomplete gamma was wrong in its
# continued-fraction branch (x/theta >= alpha+1). The Lentz coefficient was
# written `-i * (i - a)` with `i` an unsigned loop index, so the unary minus
# wrapped to 2^32 - i and the first coefficient came out as -2147483647.5
# instead of 0.5. Absolute errors reached 0.26 and the CDF exceeded 1.0 --
# Gamma(1.5, 1.0).cdf_scalar(2.5) returned 1.000498004.
#
# Separately, MAX_ITER was 100, which silently truncated the series near
# x = alpha for large shapes: alpha = 1000 was wrong by 9e-4 and alpha = 10000
# by 0.16.
#
# Both are fixed. Validated against scipy.special.gammainc over 784 points
# spanning alpha from 0.01 to 15000: worst absolute error 3.1e-12, nothing
# outside [0, 1]. These cases stay to hold that fixed.
# ---------------------------------------------------------------------------

GAMMA_SHAPES = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 3.0), (3.0, 1.0), (5.0, 2.0)]
GAMMA_X = (0.2, 0.8, 1.5, 2.5, 4.0, 10.0, 20.0)

CDF_CASES = [
    pytest.param(alpha, theta, x)
    for (alpha, theta) in GAMMA_SHAPES
    for x in GAMMA_X
]


@pytest.mark.parametrize("alpha, theta, x", CDF_CASES)
def test_cdf_matches_reference(alpha, theta, x):
    assert Gamma(alpha, theta).cdf_scalar(x) == pytest.approx(
        regularized_lower_gamma(alpha, x / theta), **ITERATIVE
    )


@pytest.mark.parametrize("alpha", [200.0, 1000.0, 5000.0])
def test_cdf_is_accurate_for_large_shape(alpha):
    """Near x = alpha the series needs ~sqrt(2*alpha*ln(1/EPS)) terms.

    At the old MAX_ITER of 100 it stopped early and returned the truncated sum
    with no indication anything was wrong.
    """
    assert Gamma(alpha, 1.0).cdf_scalar(alpha) == pytest.approx(
        regularized_lower_gamma(alpha, alpha), **ITERATIVE
    )


@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("x", [0.25, 1.0, 4.0])
def test_pdf_with_unit_shape_is_exponential(x, theta):
    """Gamma(1, th) is Exponential with rate 1/th."""
    assert Gamma(1.0, theta).pmf_scalar(x) == pytest.approx(
        math.exp(-x / theta) / theta, **EXACT
    )


@pytest.mark.parametrize("alpha, theta", PARAMS)
@pytest.mark.parametrize("x", [0.1, 2.0, 15.0])
def test_pdf_is_non_negative(x, alpha, theta):
    assert Gamma(alpha, theta).pmf_scalar(x) >= 0.0


# ---------------------------------------------------------------------------
# CDF
#
# REGRESSION: the regularized lower incomplete gamma was wrong in its
# continued-fraction branch (x/theta >= alpha+1). The Lentz coefficient was
# written `-i * (i - a)` with `i` an unsigned loop index, so the unary minus
# wrapped to 2^32 - i: the first coefficient came out as -2147483647.5 instead
# of 0.5. Absolute errors reached 0.26 and the CDF exceeded 1.0 --
# Gamma(1.5, 1.0).cdf_scalar(2.5) returned 1.000498004. alpha = 1 was correct
# throughout because that case reduces to the exact exponential form.
#
# Separately MAX_ITER was 100, which silently truncated the series near
# x = alpha for large shapes: alpha = 1000 was wrong by 9e-4, alpha = 10000 by
# 0.16.
#
# Both are fixed. Validated against scipy.special.gammainc over 784 points with
# alpha from 0.01 to 15000: worst absolute error 3.1e-12, nothing outside
# [0, 1]. These cases stay to hold that fixed.
# ---------------------------------------------------------------------------

GAMMA_SHAPES = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 3.0), (3.0, 1.0), (5.0, 2.0)]
GAMMA_X = (0.2, 0.8, 1.5, 2.5, 4.0, 10.0, 20.0)

CDF_CASES = [
    (alpha, theta, x)
    for (alpha, theta) in GAMMA_SHAPES
    for x in GAMMA_X
]


@pytest.mark.parametrize("alpha, theta, x", CDF_CASES)
def test_cdf_matches_reference(alpha, theta, x):
    assert Gamma(alpha, theta).cdf_scalar(x) == pytest.approx(
        regularized_lower_gamma(alpha, x / theta), **ITERATIVE
    )


@pytest.mark.parametrize("alpha", [200.0, 1000.0, 5000.0])
def test_cdf_is_accurate_for_large_shape(alpha):
    """Near x = alpha the series needs ~sqrt(2*alpha*ln(1/EPS)) terms.

    At the old MAX_ITER of 100 it stopped early and returned the truncated sum
    with no indication anything had gone wrong.
    """
    assert Gamma(alpha, 1.0).cdf_scalar(alpha) == pytest.approx(
        regularized_lower_gamma(alpha, alpha), **ITERATIVE
    )


@pytest.mark.parametrize("theta", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("x", [0.25, 1.0, 2.0, 6.0])
def test_cdf_with_unit_shape_is_exponential(x, theta):
    """Gamma(1, th) has the exact CDF 1 - e^(-x/th)."""
    assert Gamma(1.0, theta).cdf_scalar(x) == pytest.approx(
        1 - math.exp(-x / theta), **ITERATIVE
    )


@pytest.mark.parametrize("alpha, theta", GAMMA_SHAPES)
def test_cdf_is_bounded(alpha, theta):
    dist = Gamma(alpha, theta)
    for x in (0.1, 0.5, 1.0, 3.0, 8.0, 25.0):
        assert 0.0 <= dist.cdf_scalar(x) <= 1.0


@pytest.mark.parametrize("alpha, theta", GAMMA_SHAPES)
def test_cdf_is_monotonic(alpha, theta):
    dist = Gamma(alpha, theta)
    values = [dist.cdf_scalar(x) for x in (0.1, 0.5, 1.0, 3.0, 8.0, 25.0)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, theta", PARAMS)
def test_moments_match_closed_forms(alpha, theta):
    dist = Gamma(alpha, theta)
    assert dist.mean() == pytest.approx(alpha * theta, **EXACT)
    assert dist.variance() == pytest.approx(alpha * theta ** 2, **EXACT)
    assert dist.stddev() == pytest.approx(math.sqrt(alpha) * theta, **EXACT)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, theta", [(2.0, 3.0), (5.0, 2.0), (1.0, 1.0)])
@pytest.mark.parametrize("t_over_theta", [-1.0, -0.25, 0.0, 0.1, 0.25])
def test_mgf_matches_closed_form(t_over_theta, alpha, theta):
    t = t_over_theta / theta  # keep t strictly inside the radius of convergence
    assert Gamma(alpha, theta).mgf_scalar(t) == pytest.approx(
        gamma_mgf(t, alpha, theta), **EXACT
    )


@pytest.mark.parametrize("alpha, theta", [(2.0, 3.0), (5.0, 2.0), (1.0, 1.0)])
@pytest.mark.parametrize("t_over_theta", [-1.0, -0.25, 0.0, 0.1, 0.25])
def test_cgf_matches_closed_form(t_over_theta, alpha, theta):
    t = t_over_theta / theta
    assert Gamma(alpha, theta).cgf_scalar(t) == pytest.approx(
        gamma_cgf(t, alpha, theta), **EXACT
    )


@pytest.mark.parametrize("alpha, theta", PARAMS)
def test_mgf_at_zero_is_one(alpha, theta):
    dist = Gamma(alpha, theta)
    assert dist.mgf_scalar(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf_scalar(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("alpha, theta", [(2.0, 3.0), (5.0, 2.0)])
def test_cgf_is_log_of_mgf(alpha, theta):
    dist = Gamma(alpha, theta)
    for t in (-0.5 / theta, 0.1 / theta, 0.25 / theta):
        assert dist.cgf_scalar(t) == pytest.approx(
            math.log(dist.mgf_scalar(t)), **EXACT
        )


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, theta", PARAMS)
@pytest.mark.parametrize("x", [0.5, 3.0])
def test_classmethods_agree_with_instances(x, alpha, theta):
    dist = Gamma(alpha, theta)
    assert Gamma._pdf_scalar(x, alpha, theta) == pytest.approx(
        dist.pmf_scalar(x), **EXACT
    )
    assert Gamma._cdf_scalar(x, alpha, theta) == pytest.approx(
        dist.cdf_scalar(x), **ITERATIVE
    )
    assert Gamma._mean(alpha, theta) == pytest.approx(dist.mean(), **EXACT)
    assert Gamma._variance(alpha, theta) == pytest.approx(dist.variance(), **EXACT)
    assert Gamma._stddev(alpha, theta) == pytest.approx(dist.stddev(), **EXACT)


@pytest.mark.parametrize("method_name, args", [
    ("_pdf_scalar", (1.0, 0, 1.0)),
    ("_pdf_scalar", (1.0, 1.0, 0)),
    ("_cdf_scalar", (1.0, -1.0, 1.0)),
    ("_cdf_scalar", (1.0, 1.0, -1.0)),
    ("_mean", (0, 1.0)),
    ("_variance", (1.0, 0)),
    ("_stddev", (-2.0, 1.0)),
    ("_mgf_scalar", (0.1, 0, 1.0)),
    ("_cgf_scalar", (0.1, 1.0, -3.0)),
    ("_sample", (0, 1.0)),
])
def test_classmethods_reject_invalid_parameters(method_name, args):
    with pytest.raises(ValueError, match="must be positive"):
        getattr(Gamma, method_name)(*args)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, theta", [(1.0, 1.0), (2.0, 3.0), (5.0, 2.0)])
def test_sample_is_positive_and_finite(alpha, theta):
    dist = Gamma(alpha, theta)
    for _ in range(200):
        value = dist.sample()
        assert math.isfinite(value)
        assert value > 0.0


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Gamma(2.0, 3.0).extra = 123
