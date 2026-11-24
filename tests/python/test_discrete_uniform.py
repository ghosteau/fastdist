"""
Numeric tests for the Discrete Uniform distribution against closed-form
references.

The support is the integers a, a+1, ..., b inclusive, so there are n = b - a + 1
equally likely outcomes.
"""

import math

import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.discrete_uniform import DiscreteUniform

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1



# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def du_support_size(a: int, b: int) -> int:
    return b - a + 1


def du_mean(a: int, b: int) -> float:
    return (a + b) / 2.0


def du_variance(a: int, b: int) -> float:
    """Var = ((b - a + 1)^2 - 1) / 12"""
    n = du_support_size(a, b)
    return (n ** 2 - 1) / 12.0


def du_mgf(t: float, a: int, b: int) -> float:
    """M(t) = (1/n) sum_{k=a}^{b} e^(tk); equals 1 at t = 0."""
    n = du_support_size(a, b)
    return sum(math.exp(t * k) for k in range(a, b + 1)) / n


PARAMS = [(1, 6), (0, 1), (-3, 3), (2, 10), (-5, -1)]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    dist = DiscreteUniform(a=1, b=6)
    assert dist.a == 1
    assert dist.b == 6


@pytest.mark.parametrize("a, b", [(5, 5), (6, 1), (0, -1)])
def test_init_rejects_non_increasing_bounds(a, b):
    with pytest.raises(ValueError, match="a must be less than b"):
        DiscreteUniform(a=a, b=b)


@pytest.mark.parametrize("bad", [1.5, "1"])
def test_init_rejects_non_integer_bounds(bad):
    with pytest.raises(TypeError, match="a must be an integer"):
        DiscreteUniform(a=bad, b=10)
    with pytest.raises(TypeError, match="b must be an integer"):
        DiscreteUniform(a=1, b=bad)


def test_repr():
    assert repr(DiscreteUniform(a=1, b=6)) == "DiscreteUniform(a=1, b=6)"


# REGRESSION: the b setter assigned `self.b = value` instead of `self._b`, so
# it re-entered itself and raised RecursionError for every assignment.
def test_b_setter_updates_value():
    dist = DiscreteUniform(a=1, b=6)
    dist.b = 10
    assert dist.b == 10


# REGRESSION: the a setter stored float(value) even though a is an integer
# parameter that __init__ stores via int(), so assigning through the setter
# silently changed the attribute's type.
def test_a_setter_preserves_integer_type():
    dist = DiscreteUniform(a=1, b=6)
    dist.a = 2
    assert dist.a == 2
    assert isinstance(dist.a, int)


def test_a_setter_validates():
    dist = DiscreteUniform(a=1, b=6)
    with pytest.raises(TypeError, match="a must be an integer"):
        dist.a = 1.5


# ---------------------------------------------------------------------------
# PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_pmf_is_uniform_over_the_support(a, b):
    dist = DiscreteUniform(a, b)
    expected = 1.0 / du_support_size(a, b)
    for k in range(a, b + 1):
        assert dist.pmf(k) == pytest.approx(expected, **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_pmf_sums_to_one(a, b):
    dist = DiscreteUniform(a, b)
    total = sum(dist.pmf(k) for k in range(a, b + 1))
    assert total == pytest.approx(1.0, **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_pmf_is_zero_outside_the_support(a, b):
    dist = DiscreteUniform(a, b)
    assert dist.pmf(a - 1) == pytest.approx(0.0, abs=1e-15)
    assert dist.pmf(b + 1) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_matches_closed_form(a, b):
    """F(k) = (k - a + 1) / n on the support."""
    dist = DiscreteUniform(a, b)
    n = du_support_size(a, b)
    for k in range(a, b + 1):
        assert dist.cdf(k) == pytest.approx((k - a + 1) / n, **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_matches_summed_pmf(a, b):
    dist = DiscreteUniform(a, b)
    for k in range(a, b + 1):
        assert dist.cdf(k) == pytest.approx(
            sum(dist.pmf(i) for i in range(a, k + 1)), **ITERATIVE
        )


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_is_monotonic_and_bounded(a, b):
    dist = DiscreteUniform(a, b)
    values = [dist.cdf(k) for k in range(a - 2, b + 3)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_saturates_at_the_bounds(a, b):
    dist = DiscreteUniform(a, b)
    assert dist.cdf(a - 1) == pytest.approx(0.0, abs=1e-15)
    assert dist.cdf(b) == pytest.approx(1.0, **EXACT)
    assert dist.cdf(b + 5) == pytest.approx(1.0, **EXACT)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_moments_match_closed_forms(a, b):
    dist = DiscreteUniform(a, b)
    assert dist.mean() == pytest.approx(du_mean(a, b), **EXACT)
    assert dist.variance() == pytest.approx(du_variance(a, b), **EXACT)
    assert dist.stddev() == pytest.approx(math.sqrt(du_variance(a, b)), **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_mean_equals_pmf_weighted_sum(a, b):
    dist = DiscreteUniform(a, b)
    expectation = sum(k * dist.pmf(k) for k in range(a, b + 1))
    assert dist.mean() == pytest.approx(expectation, **ITERATIVE)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
@pytest.mark.parametrize("t", [-0.5, -0.1, 0.1, 0.5])
def test_mgf_matches_closed_form(t, a, b):
    assert DiscreteUniform(a, b).mgf(t) == pytest.approx(du_mgf(t, a, b), **ITERATIVE)


@pytest.mark.parametrize("a, b", PARAMS)
def test_mgf_at_zero_is_one(a, b):
    """The t = 0 case is a removable 0/0 singularity in the usual closed form."""
    dist = DiscreteUniform(a, b)
    assert dist.mgf(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a, b", PARAMS)
@pytest.mark.parametrize("t", [-0.5, -0.1, 0.1, 0.5])
def test_cgf_is_log_of_mgf(t, a, b):
    dist = DiscreteUniform(a, b)
    assert dist.cgf(t) == pytest.approx(math.log(dist.mgf(t)), **ITERATIVE)


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_classmethods_agree_with_instances(a, b):
    dist = DiscreteUniform(a, b)
    assert DiscreteUniform._pmf_scalar(a, a, b) == pytest.approx(
        dist.pmf(a), **EXACT
    )
    assert DiscreteUniform._cdf_scalar(a, a, b) == pytest.approx(
        dist.cdf(a), **EXACT
    )
    assert DiscreteUniform._mean(a, b) == pytest.approx(dist.mean(), **EXACT)
    assert DiscreteUniform._variance(a, b) == pytest.approx(dist.variance(), **EXACT)
    assert DiscreteUniform._stddev(a, b) == pytest.approx(dist.stddev(), **EXACT)
    assert DiscreteUniform._mgf_scalar(0.1, a, b) == pytest.approx(
        dist.mgf(0.1), **ITERATIVE
    )
    assert DiscreteUniform._cgf_scalar(0.1, a, b) == pytest.approx(
        dist.cgf(0.1), **ITERATIVE
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, 6, 1)),
    ("_cdf_scalar", (1, 6, 1)),
    ("_mean", (6, 1)),
    ("_variance", (5, 5)),
    ("_stddev", (10, 2)),
    ("_mgf_scalar", (0.1, 6, 1)),
    ("_cgf_scalar", (0.1, 6, 1)),
    ("_sample", (6, 1)),
])
def test_classmethods_reject_non_increasing_bounds(method_name, args):
    with pytest.raises(ValueError, match="a must be less than b"):
        getattr(DiscreteUniform, method_name)(*args)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [(1, 6), (-3, 3), (0, 1)])
def test_sample_lies_within_the_support(a, b):
    dist = DiscreteUniform(a, b)
    for _ in range(300):
        value = dist.sample()
        assert isinstance(value, int)
        assert a <= value <= b


def test_sample_eventually_covers_the_whole_support():
    """With 2000 draws from a 6-point support, every value should appear."""
    dist = DiscreteUniform(1, 6)
    seen = {dist.sample() for _ in range(2000)}
    assert seen == {1, 2, 3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        DiscreteUniform(1, 6).extra = 123


# ---------------------------------------------------------------------------
# Numerical edge cases
#
# Regressions for confirmed defects. Each docstring states what the backend
# did before the fix, and every expected value is derived independently -- from
# a closed form, exact rational arithmetic, or a numerically stable
# rearrangement -- rather than captured from a previous run.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t, a, b", [(1e-10, 0, 100), (1e-12, 5, 20), (1e-8, 0, 10)])
def test_mgf_does_not_cancel_for_small_t(t, a, b):
    """
    Written as (e^t)^n - 1 over e^t - 1, both subtractions cancel for small t
    and the power squares the rounding error on top. expm1 on each is exact.
    """
    n = b - a + 1
    got = DiscreteUniform(a, b).mgf(t)
    want = math.exp(t * a) * math.expm1(t * n) / (n * math.expm1(t))
    assert got == pytest.approx(want, rel=1e-15)


def test_pmf_over_the_full_int_range():
    """
    The support size was formed as b - a + 1 in int, which wraps for any range
    wider than INT_MAX. Over the full int range it wrapped to 0 and the PMF
    came back infinite.
    """
    got = DiscreteUniform(INT_MIN, INT_MAX).pmf(0)
    assert got == pytest.approx(1.0 / 2**32, rel=1e-15)


def test_variance_over_the_full_int_range_is_positive():
    """The same wrap made the variance negative, which no caller would expect."""
    got = DiscreteUniform(INT_MIN, INT_MAX).variance()
    assert got > 0.0, "a variance must never be negative"
    assert got == pytest.approx((2.0**64 - 1) / 12, rel=1e-15)


def test_stddev_over_the_full_int_range():
    """Follows the variance, so it was wrong in the same way."""
    got = DiscreteUniform(INT_MIN, INT_MAX).stddev()
    assert got == pytest.approx(math.sqrt((2.0**64 - 1) / 12), rel=1e-15)


def test_cdf_over_the_full_int_range():
    """The median of the full int range sits at one half."""
    got = DiscreteUniform(INT_MIN, INT_MAX).cdf(-1)
    assert got == pytest.approx(0.5, rel=1e-15)
