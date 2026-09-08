"""
Numeric tests for the Negative Binomial distribution against closed-form
references.

This implementation uses the "number of failures before the r-th success"
convention: the support is k = 0, 1, 2, ... with
P(X = k) = C(k + r - 1, k) p^r (1-p)^k, so the mean is r(1-p)/p.
"""

import math

import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.negative_binomial import NegativeBinomial


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def nbinom_pmf(k: int, r: int, p: float) -> float:
    """P(X = k) = C(k + r - 1, k) p^r (1-p)^k for k >= 0"""
    return math.comb(k + r - 1, k) * p ** r * (1 - p) ** k


def nbinom_cdf(k: int, r: int, p: float) -> float:
    return sum(nbinom_pmf(i, r, p) for i in range(0, k + 1))


def nbinom_mgf(t: float, r: int, p: float) -> float:
    """M(t) = (p / (1 - (1-p) e^t))^r, valid for t < -ln(1-p)"""
    return (p / (1 - (1 - p) * math.exp(t))) ** r


PARAMS = [(1, 0.5), (3, 0.5), (2, 0.3), (5, 0.7), (4, 0.9)]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    dist = NegativeBinomial(r=3, p=0.5)
    assert dist.r == 3
    assert dist.p == 0.5


@pytest.mark.parametrize("r", [0, -1, -10])
def test_init_rejects_non_positive_r(r):
    with pytest.raises(ValueError, match="r must be positive"):
        NegativeBinomial(r=r, p=0.5)


@pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
def test_init_rejects_out_of_range_p(p):
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        NegativeBinomial(r=3, p=p)


@pytest.mark.parametrize("bad_r", [2.5, "3"])
def test_init_rejects_non_integer_r(bad_r):
    with pytest.raises(TypeError, match="r must be an integer"):
        NegativeBinomial(r=bad_r, p=0.5)


@pytest.mark.parametrize("bad_p", ["0.5", object()])
def test_init_rejects_non_real_p(bad_p):
    with pytest.raises(TypeError, match="p must be a real number"):
        NegativeBinomial(r=3, p=bad_p)


def test_repr():
    assert repr(NegativeBinomial(r=3, p=0.5)) == "NegativeBinomial(r=3, p=0.5)"


def test_property_setters_update_and_validate():
    dist = NegativeBinomial(r=3, p=0.5)
    dist.r = 5
    dist.p = 0.7
    assert dist.r == 5
    assert dist.p == 0.7
    with pytest.raises(ValueError, match="r must be positive"):
        dist.r = 0
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        dist.p = 1.5


# ---------------------------------------------------------------------------
# PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, p", PARAMS)
@pytest.mark.parametrize("k", [0, 1, 2, 3, 5, 10])
def test_pmf_matches_closed_form(k, r, p):
    assert NegativeBinomial(r, p).pmf_scalar(k) == pytest.approx(
        nbinom_pmf(k, r, p), **EXACT
    )


# The summation stops at k = 160 rather than running to convergence because the
# PMF overflows to inf/nan beyond that point; see the overflow section below.
# Every parameter set here has negligible mass past k = 160 (< 1e-20).
SUMMATION_LIMIT = 160


@pytest.mark.parametrize("r, p", PARAMS)
def test_pmf_sums_to_one(r, p):
    dist = NegativeBinomial(r, p)
    total = sum(dist.pmf_scalar(k) for k in range(0, SUMMATION_LIMIT))
    assert total == pytest.approx(1.0, **ITERATIVE)


@pytest.mark.parametrize("r, p", PARAMS)
@pytest.mark.parametrize("k", [0, 2, 7])
def test_pmf_is_a_probability(k, r, p):
    assert 0.0 <= NegativeBinomial(r, p).pmf_scalar(k) <= 1.0


def test_pmf_with_unit_r_is_geometric_on_failures():
    """NegativeBinomial(1, p) gives P(X = k) = p (1-p)^k."""
    dist = NegativeBinomial(1, 0.3)
    for k in range(10):
        assert dist.pmf_scalar(k) == pytest.approx(0.3 * 0.7 ** k, **EXACT)


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, p", PARAMS)
@pytest.mark.parametrize("k", [0, 1, 3, 6, 12])
def test_cdf_matches_summed_pmf(k, r, p):
    assert NegativeBinomial(r, p).cdf_scalar(k) == pytest.approx(
        nbinom_cdf(k, r, p), **ITERATIVE
    )


@pytest.mark.parametrize("r, p", PARAMS)
def test_cdf_is_monotonic_and_bounded(r, p):
    dist = NegativeBinomial(r, p)
    values = [dist.cdf_scalar(k) for k in range(0, 25)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("r, p", [(3, 0.5), (5, 0.7), (4, 0.9)])
def test_cdf_approaches_one_in_the_tail(r, p):
    assert NegativeBinomial(r, p).cdf_scalar(150) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Numeric range
#
# REGRESSION: the PMF used to evaluate C(k + r - 1, k) from raw factorials, so
# the intermediate (k + r - 1)! overflowed a double once k + r - 1 > 170 --
# inf at k = 170, nan from k = 200 onward, and cdf_scalar(200) nan with it.
#
# The coefficient itself is small: for r = 3, k = 200 it is C(202, 200) = 20301
# and the true PMF is 1.58e-57, comfortably inside double range. Evaluating the
# whole PMF in log space via lgamma keeps the intermediates small and removes
# the ceiling entirely. These cases stay to hold that fixed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [170, 200, 500, 1000])
def test_pmf_stays_finite_in_the_far_tail(k):
    value = NegativeBinomial(3, 0.5).pmf_scalar(k)
    assert math.isfinite(value)
    assert value >= 0.0


@pytest.mark.parametrize("k", [200, 500])
def test_pmf_matches_closed_form_in_the_far_tail(k):
    assert NegativeBinomial(3, 0.5).pmf_scalar(k) == pytest.approx(
        nbinom_pmf(k, 3, 0.5), **ITERATIVE
    )


@pytest.mark.parametrize("k", [200, 500])
def test_cdf_stays_finite_in_the_far_tail(k):
    assert NegativeBinomial(3, 0.5).cdf_scalar(k) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, p", PARAMS)
def test_moments_match_closed_forms(r, p):
    dist = NegativeBinomial(r, p)
    assert dist.mean() == pytest.approx(r * (1 - p) / p, **EXACT)
    assert dist.variance() == pytest.approx(r * (1 - p) / p ** 2, **EXACT)
    assert dist.stddev() == pytest.approx(
        math.sqrt(r * (1 - p) / p ** 2), **EXACT
    )


@pytest.mark.parametrize("r, p", [(3, 0.5), (2, 0.3), (5, 0.7)])
def test_mean_equals_pmf_weighted_sum(r, p):
    dist = NegativeBinomial(r, p)
    expectation = sum(k * dist.pmf_scalar(k) for k in range(0, SUMMATION_LIMIT))
    assert dist.mean() == pytest.approx(expectation, rel=1e-8)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, p", [(3, 0.5), (2, 0.3), (5, 0.7)])
@pytest.mark.parametrize("t", [-1.0, -0.2, 0.0, 0.1])
def test_mgf_matches_closed_form(t, r, p):
    assert NegativeBinomial(r, p).mgf_scalar(t) == pytest.approx(
        nbinom_mgf(t, r, p), **EXACT
    )


@pytest.mark.parametrize("r, p", [(3, 0.5), (2, 0.3), (5, 0.7)])
@pytest.mark.parametrize("t", [-1.0, -0.2, 0.0, 0.1])
def test_cgf_matches_closed_form(t, r, p):
    assert NegativeBinomial(r, p).cgf_scalar(t) == pytest.approx(
        math.log(nbinom_mgf(t, r, p)), **EXACT
    )


@pytest.mark.parametrize("r, p", PARAMS)
def test_mgf_at_zero_is_one(r, p):
    dist = NegativeBinomial(r, p)
    assert dist.mgf_scalar(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf_scalar(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("r, p", [(3, 0.5), (5, 0.7)])
def test_cgf_is_log_of_mgf(r, p):
    dist = NegativeBinomial(r, p)
    for t in (-0.5, -0.1, 0.05):
        assert dist.cgf_scalar(t) == pytest.approx(
            math.log(dist.mgf_scalar(t)), **EXACT
        )


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, p", PARAMS)
def test_classmethods_agree_with_instances(r, p):
    dist = NegativeBinomial(r, p)
    assert NegativeBinomial._pmf_scalar(3, r, p) == pytest.approx(
        dist.pmf_scalar(3), **EXACT
    )
    assert NegativeBinomial._cdf_scalar(3, r, p) == pytest.approx(
        dist.cdf_scalar(3), **ITERATIVE
    )
    assert NegativeBinomial._mean(r, p) == pytest.approx(dist.mean(), **EXACT)
    assert NegativeBinomial._variance(r, p) == pytest.approx(
        dist.variance(), **EXACT
    )
    assert NegativeBinomial._stddev(r, p) == pytest.approx(dist.stddev(), **EXACT)
    assert NegativeBinomial._mgf_scalar(0.1, r, p) == pytest.approx(
        dist.mgf_scalar(0.1), **EXACT
    )
    assert NegativeBinomial._cgf_scalar(0.1, r, p) == pytest.approx(
        dist.cgf_scalar(0.1), **EXACT
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, 0, 0.5)),
    ("_cdf_scalar", (1, -2, 0.5)),
    ("_mean", (0, 0.5)),
    ("_variance", (-1, 0.5)),
    ("_stddev", (0, 0.5)),
    ("_mgf_scalar", (0.1, 0, 0.5)),
    ("_cgf_scalar", (0.1, -3, 0.5)),
    ("_sample", (0, 0.5)),
])
def test_classmethods_reject_non_positive_r(method_name, args):
    with pytest.raises(ValueError, match="r must be positive"):
        getattr(NegativeBinomial, method_name)(*args)


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, 3, -0.1)),
    ("_cdf_scalar", (1, 3, 1.5)),
    ("_mean", (3, 2.0)),
    ("_variance", (3, -1.0)),
    ("_sample", (3, 1.7)),
])
def test_classmethods_reject_out_of_range_p(method_name, args):
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        getattr(NegativeBinomial, method_name)(*args)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("r, p", [(3, 0.5), (2, 0.3), (5, 0.9)])
def test_sample_lies_within_the_support(r, p):
    dist = NegativeBinomial(r, p)
    for _ in range(300):
        value = dist.sample()
        assert isinstance(value, int)
        assert value >= 0


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        NegativeBinomial(3, 0.5).extra = 123
