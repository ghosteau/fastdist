"""Numeric tests for the Beta distribution against closed-form references."""

import math

import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.beta import Beta


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def beta_fn(a: float, b: float) -> float:
    """B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b)"""
    return math.gamma(a) * math.gamma(b) / math.gamma(a + b)


def beta_pdf(x: float, a: float, b: float) -> float:
    """f(x; a, b) = x^(a-1) (1-x)^(b-1) / B(a, b)"""
    return x ** (a - 1) * (1 - x) ** (b - 1) / beta_fn(a, b)


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def beta_variance(a: float, b: float) -> float:
    return a * b / ((a + b) ** 2 * (a + b + 1))


PARAMS = [(2.0, 3.0), (1.0, 1.0), (0.5, 0.5), (5.0, 2.0), (3.0, 3.0)]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    dist = Beta(alpha=2.0, beta=3.0)
    assert dist.alpha == 2.0
    assert dist.beta == 3.0


@pytest.mark.parametrize("alpha, beta", [(0, 1.0), (-1, 1.0), (1.0, 0), (1.0, -1)])
def test_init_rejects_non_positive_parameters(alpha, beta):
    with pytest.raises(ValueError, match="must be positive"):
        Beta(alpha=alpha, beta=beta)


@pytest.mark.parametrize("bad", ["2.0", None, object()])
def test_init_rejects_non_real_parameters(bad):
    with pytest.raises((TypeError, ValueError)):
        Beta(alpha=bad, beta=1.0)


def test_repr():
    assert repr(Beta(alpha=2.0, beta=3.0)) == "Beta(alpha=2.0, beta=3.0)"


def test_alpha_setter_updates_and_validates():
    dist = Beta(alpha=2.0, beta=3.0)
    dist.alpha = 4.0
    assert dist.alpha == 4.0
    with pytest.raises(ValueError, match="alpha must be positive"):
        dist.alpha = -1.0


# KNOWN BUG: the beta setter calls _validate_params(beta=value), leaving alpha
# as None. Because the `if alpha <= 0` check sits outside the `if alpha is not
# None` guard (beta.py line 47), the comparison None <= 0 raises TypeError for
# *every* assignment, valid or not. The beta property is unusable.
@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason="beta setter raises TypeError for any value")
def test_beta_setter_updates_value():
    dist = Beta(alpha=2.0, beta=3.0)
    dist.beta = 5.0
    assert dist.beta == 5.0


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason="beta setter raises TypeError before validating")
def test_beta_setter_rejects_non_positive():
    dist = Beta(alpha=2.0, beta=3.0)
    with pytest.raises(ValueError, match="beta must be positive"):
        dist.beta = 0


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, beta", PARAMS)
@pytest.mark.parametrize("x", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_pdf_matches_closed_form(x, alpha, beta):
    assert Beta(alpha, beta).pdf_scalar(x) == pytest.approx(
        beta_pdf(x, alpha, beta), **EXACT
    )


@pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
def test_pdf_of_uniform_special_case_is_one(x):
    """Beta(1, 1) is the uniform distribution on [0, 1]."""
    assert Beta(1.0, 1.0).pdf_scalar(x) == pytest.approx(1.0, **EXACT)


@pytest.mark.parametrize("alpha, beta", PARAMS)
@pytest.mark.parametrize("x", [0.05, 0.4, 0.95])
def test_pdf_is_non_negative(x, alpha, beta):
    assert Beta(alpha, beta).pdf_scalar(x) >= 0.0


@pytest.mark.parametrize("alpha, beta", [(2.0, 3.0), (5.0, 2.0)])
def test_pdf_is_symmetric_under_parameter_swap(alpha, beta):
    """f(x; a, b) == f(1-x; b, a)"""
    assert Beta(alpha, beta).pdf_scalar(0.3) == pytest.approx(
        Beta(beta, alpha).pdf_scalar(0.7), **EXACT
    )


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, beta", PARAMS)
def test_moments_match_closed_forms(alpha, beta):
    dist = Beta(alpha, beta)
    assert dist.mean() == pytest.approx(beta_mean(alpha, beta), **EXACT)
    assert dist.variance() == pytest.approx(beta_variance(alpha, beta), **EXACT)
    assert dist.stddev() == pytest.approx(
        math.sqrt(beta_variance(alpha, beta)), **EXACT
    )


@pytest.mark.parametrize("alpha, beta", PARAMS)
def test_mean_lies_inside_the_support(alpha, beta):
    assert 0.0 < Beta(alpha, beta).mean() < 1.0


# ---------------------------------------------------------------------------
# CDF
#
# KNOWN BUG: the regularized incomplete beta in src/math/beta.cpp is incorrect.
# It disagrees with numerical integration at every tested point, returns 0.6534
# for Beta(1, 1) at x=0.5 where the exact answer is 0.5, and returns a negative
# value (-0.5958) for Beta(0.5, 0.5) at x=0.5, which is impossible for a CDF.
# These tests assert the correct behaviour and are marked strict-xfail so they
# turn into failures the moment the backend is fixed and the marker goes stale.
# ---------------------------------------------------------------------------

@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason="beta_cdf_scalar is numerically incorrect")
@pytest.mark.parametrize("x", [0.1, 0.3, 0.5, 0.75, 0.9])
def test_cdf_of_uniform_special_case_is_identity(x):
    """Beta(1, 1) is uniform on [0, 1], so its CDF is F(x) = x exactly."""
    assert Beta(1.0, 1.0).cdf_scalar(x) == pytest.approx(x, **ITERATIVE)


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason="beta_cdf_scalar is numerically incorrect")
def test_cdf_matches_analytic_integral():
    """For Beta(2, 3), F(0.5) = 0.6875 by direct integration of 12x(1-x)^2."""
    assert Beta(2.0, 3.0).cdf_scalar(0.5) == pytest.approx(0.6875, **ITERATIVE)


# The CDF is wrong everywhere, but it still happens to be bounded and monotonic
# for most parameters. It breaks both properties only for alpha = beta = 0.5,
# where it returns values as low as -0.4273. Only that case is xfailed, so the
# structural guarantees stay enforced for every other parameter pair.
CDF_PROPERTY_PARAMS = [
    (2.0, 3.0),
    (1.0, 1.0),
    pytest.param(
        0.5, 0.5,
        marks=[
            pytest.mark.known_bug,
            pytest.mark.xfail(
                strict=True,
                reason="beta_cdf_scalar returns negative values for alpha=beta=0.5",
            ),
        ],
    ),
    (5.0, 2.0),
    (3.0, 3.0),
]


@pytest.mark.parametrize("alpha, beta", CDF_PROPERTY_PARAMS)
def test_cdf_is_bounded(alpha, beta):
    dist = Beta(alpha, beta)
    for x in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert 0.0 <= dist.cdf_scalar(x) <= 1.0


@pytest.mark.parametrize("alpha, beta", CDF_PROPERTY_PARAMS)
def test_cdf_is_monotonic(alpha, beta):
    dist = Beta(alpha, beta)
    values = [dist.cdf_scalar(x) for x in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)]
    assert values == sorted(values)


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason="beta_cdf_scalar is numerically incorrect")
def test_cdf_is_symmetric_for_symmetric_parameters():
    """For a == b the distribution is symmetric about 0.5, so F(0.5) == 0.5."""
    assert Beta(3.0, 3.0).cdf_scalar(0.5) == pytest.approx(0.5, **ITERATIVE)


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, beta", PARAMS)
@pytest.mark.parametrize("x", [0.25, 0.5])
def test_classmethods_agree_with_instances(x, alpha, beta):
    dist = Beta(alpha, beta)
    assert Beta._pdf_scalar(x, alpha, beta) == pytest.approx(
        dist.pdf_scalar(x), **EXACT
    )
    assert Beta._mean(alpha, beta) == pytest.approx(dist.mean(), **EXACT)
    assert Beta._variance(alpha, beta) == pytest.approx(dist.variance(), **EXACT)
    assert Beta._stddev(alpha, beta) == pytest.approx(dist.stddev(), **EXACT)


@pytest.mark.parametrize("method_name, args", [
    ("_pdf_scalar", (0.5, -0.1, 1.0)),
    ("_pdf_scalar", (0.5, 1.0, -0.1)),
    ("_pdf_scalar", (0.5, 0, 1.0)),
    ("_pdf_scalar", (0.5, 1.0, 0)),
    ("_cdf_scalar", (0.5, -1, 1.0)),
    ("_cdf_scalar", (0.5, 1.0, -1)),
    ("_mean", (0, 1.0)),
    ("_variance", (1.0, -2.0)),
    ("_stddev", (-3.0, 1.0)),
    ("_sample", (0, 1.0)),
])
def test_classmethods_reject_invalid_parameters(method_name, args):
    with pytest.raises(ValueError, match="must be positive"):
        getattr(Beta, method_name)(*args)


# ---------------------------------------------------------------------------
# Validation edge case
#
# KNOWN BUG: in Beta._validate_params the `if alpha <= 0` check sits outside the
# `if alpha is not None` guard (beta.py line 47), so validating only `beta`
# raises TypeError comparing None to int.
# ---------------------------------------------------------------------------

@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason="alpha <= 0 check sits outside the None guard")
def test_validate_params_accepts_a_single_named_parameter():
    Beta._validate_params(beta=3.0)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha, beta", [(2.0, 3.0), (1.0, 1.0), (5.0, 2.0)])
def test_sample_lies_within_the_unit_interval(alpha, beta):
    dist = Beta(alpha, beta)
    for _ in range(200):
        value = dist.sample()
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Beta(2.0, 3.0).extra = 123
