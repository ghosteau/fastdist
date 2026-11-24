"""
Regression tests for defects found auditing the C++ core.

Each test here corresponds to a confirmed bug: the behaviour described in the
docstring is what the backend did before the fix. They are grouped by the kind
of mistake rather than by distribution, because the same mistake turned up in
several places and the grouping is what stops it being reintroduced.

Every expected value is derived independently -- from an exact rational
computation, a closed form, or a numerically stable rearrangement -- never
captured from a previous run of this library.
"""

import math
from fractions import Fraction

import pytest

from fastdist import _fastdist as core
from fastdist.distributions.beta import Beta
from fastdist.distributions.discrete_uniform import DiscreteUniform
from fastdist.distributions.geometric import Geometric

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# Catastrophic cancellation
#
# Each of these subtracts two nearly equal quantities. The stable form is
# algebraically identical, so the test compares against a rearrangement that
# cannot cancel rather than against a recorded number.
# ---------------------------------------------------------------------------

class TestCancellation:
    @pytest.mark.parametrize("k, p", [(1, 1e-10), (3, 1e-12), (1, 1e-15), (2, 1e-8), (5, 1e-9)])
    def test_geometric_cdf_small_p(self, k, p):
        """1 - (1-p)^k lost up to three of four significant digits for tiny p."""
        got = core.geometric_cdf_scalar(k, p)
        want = -math.expm1(k * math.log1p(-p))
        assert got == pytest.approx(want, rel=1e-15)

    def test_geometric_cdf_matches_exact_rational(self):
        """F(k) = 1 - (1-p)^k, computed exactly in rational arithmetic."""
        p = Fraction(1, 10**12)
        k = 3
        exact = 1 - (1 - p) ** k
        got = core.geometric_cdf_scalar(k, float(p))
        assert abs(Fraction(got) - exact) / exact < 1e-15

    @pytest.mark.parametrize("t, a, b", [(1e-9, 0.0, 1.0), (1e-11, 0.0, 1.0),
                                         (1e-13, 2.0, 5.0), (1e-7, 0.0, 1.0)])
    def test_uniform_mgf_small_t(self, t, a, b):
        """exp(bt) - exp(at) cancels for small t; expm1 of each term does not."""
        got = core.uniform_mgf_scalar(t, a, b)
        want = (math.expm1(b * t) - math.expm1(a * t)) / (t * (b - a))
        assert got == pytest.approx(want, rel=1e-15)

    @pytest.mark.parametrize("t, a, b", [(1e-10, 0, 100), (1e-12, 5, 20), (1e-8, 0, 10)])
    def test_discrete_uniform_mgf_small_t(self, t, a, b):
        """(e^t)^n - 1 over e^t - 1 cancels twice over for small t."""
        n = b - a + 1
        got = core.discrete_uniform_mgf_scalar(t, a, b)
        want = math.exp(t * a) * math.expm1(t * n) / (n * math.expm1(t))
        assert got == pytest.approx(want, rel=1e-15)


# ---------------------------------------------------------------------------
# Integer overflow
#
# b - a + 1 was formed in int, so any range wider than INT_MAX wrapped. The
# PMF came back infinite and the variance negative.
# ---------------------------------------------------------------------------

class TestIntegerOverflow:
    def test_discrete_uniform_pmf_full_int_range(self):
        got = core.discrete_uniform_pmf_scalar(0, INT_MIN, INT_MAX)
        assert got == pytest.approx(1.0 / 2**32, rel=1e-15)

    def test_discrete_uniform_variance_full_int_range_is_positive(self):
        got = core.discrete_uniform_variance(INT_MIN, INT_MAX)
        assert got > 0.0, "a variance must never be negative"
        assert got == pytest.approx((2.0**64 - 1) / 12, rel=1e-15)

    def test_discrete_uniform_cdf_full_int_range(self):
        """The median of the full int range sits at one half."""
        got = core.discrete_uniform_cdf_scalar(-1, INT_MIN, INT_MAX)
        assert got == pytest.approx(0.5, rel=1e-15)

    def test_discrete_uniform_stddev_full_int_range(self):
        got = core.discrete_uniform_stddev(INT_MIN, INT_MAX)
        assert got == pytest.approx(math.sqrt((2.0**64 - 1) / 12), rel=1e-15)

    @pytest.mark.parametrize("x", [2.0**31, 2.0**31 + 1, 3e9, 1e18])
    def test_poisson_cdf_beyond_int_max(self, x):
        """
        static_cast<int> of a double past INT_MAX is undefined; it wrapped to
        INT_MIN, skipped the summation loop and returned exp(-lambda).
        Far above the mean the CDF is 1.
        """
        assert core.poisson_cdf_scalar(x, 5.0) == 1.0


# ---------------------------------------------------------------------------
# Unbounded loops
#
# These summations ran to their argument even once every remaining term was
# too small to change the result.
# ---------------------------------------------------------------------------

class TestLoopTermination:
    def test_poisson_cdf_huge_argument_is_fast_and_correct(self):
        import time
        t0 = time.perf_counter()
        got = core.poisson_cdf_scalar(2_000_000_000.0, 5.0)
        elapsed = time.perf_counter() - t0
        assert got == 1.0
        assert elapsed < 1.0, f"took {elapsed:.2f}s; the tail sum is not terminating early"

    def test_negative_binomial_cdf_huge_argument_is_fast_and_correct(self):
        import time
        t0 = time.perf_counter()
        got = core.negative_binomial_cdf_scalar(200_000_000, 5, 0.5)
        elapsed = time.perf_counter() - t0
        assert got == 1.0
        assert elapsed < 1.0, f"took {elapsed:.2f}s; the tail sum is not terminating early"

    def test_factorial_past_overflow_returns_inf_immediately(self):
        import time
        t0 = time.perf_counter()
        got = core.factorial(10_000_000)
        elapsed = time.perf_counter() - t0
        assert math.isinf(got)
        assert elapsed < 0.5, f"took {elapsed:.2f}s; it is still multiplying past infinity"

    def test_factorial_boundary(self):
        """170! is the largest representable factorial; 171! overflows."""
        assert math.isfinite(core.factorial(170))
        assert math.isinf(core.factorial(171))
        assert core.factorial(20) == pytest.approx(float(math.factorial(20)), rel=1e-15)


# ---------------------------------------------------------------------------
# CGF computed as log(MGF)
#
# The MGF raises a base to a large power and overflows to infinity, so log of
# it reported NaN for a cumulant that is an ordinary number.
# ---------------------------------------------------------------------------

class TestCgfOverflow:
    @pytest.mark.parametrize("t, alpha, theta", [(0.99, 1000.0, 1.0), (0.5, 2000.0, 1.0),
                                                 (0.9, 5000.0, 1.0)])
    def test_gamma_cgf_large_alpha(self, t, alpha, theta):
        got = core.gamma_cgf_scalar(t, alpha, theta)
        want = -alpha * math.log1p(-theta * t)
        assert math.isfinite(got), "the MGF overflowed and the CGF reported NaN"
        assert got == pytest.approx(want, rel=1e-14)

    @pytest.mark.parametrize("t, r, p", [(0.5, 1000, 0.5), (0.6, 2000, 0.5), (0.1, 5000, 0.5)])
    def test_negative_binomial_cgf_large_r(self, t, r, p):
        got = core.negative_binomial_cgf_scalar(t, r, p)
        want = r * (math.log(p) - math.log1p(-(1 - p) * math.exp(t)))
        assert math.isfinite(got), "the MGF overflowed and the CGF reported NaN"
        assert got == pytest.approx(want, rel=1e-14)

    def test_cgf_is_log_of_mgf_where_both_are_representable(self):
        """Where the MGF does not overflow, the two routes must still agree."""
        for t, alpha in [(0.1, 2.0), (-0.5, 3.0), (0.25, 10.0)]:
            mgf = core.gamma_mgf_scalar(t, alpha, 1.0)
            cgf = core.gamma_cgf_scalar(t, alpha, 1.0)
            assert cgf == pytest.approx(math.log(mgf), rel=1e-14)


# ---------------------------------------------------------------------------
# Missing parameter validation
#
# Every entry point is supposed to return NaN for parameters outside its
# domain. These returned a plausible-looking number instead.
# ---------------------------------------------------------------------------

class TestValidationGaps:
    @pytest.mark.parametrize("alpha, beta", [(-1.0, -2.0), (-5.0, 3.0), (0.0, 1.0),
                                             (1.0, 0.0), (float("nan"), 1.0),
                                             (float("inf"), 1.0), (1.0, float("-inf"))])
    def test_beta_moments_reject_invalid_shapes(self, alpha, beta):
        """beta_variance(-1, -2) returned -0.111: a negative variance."""
        assert math.isnan(core.beta_mean(alpha, beta))
        assert math.isnan(core.beta_variance(alpha, beta))
        assert math.isnan(core.beta_stddev(alpha, beta))

    def test_beta_moments_still_correct_for_valid_shapes(self):
        assert core.beta_mean(2.0, 3.0) == pytest.approx(0.4, rel=1e-15)
        assert core.beta_variance(2.0, 3.0) == pytest.approx(2 * 3 / (25 * 6), rel=1e-15)

    @pytest.mark.parametrize("x, mu, sigma", [(1.0, 0.0, 0.0), (1.0, 0.0, -2.0),
                                              (float("nan"), 0.0, 1.0),
                                              (1.0, float("inf"), 1.0)])
    def test_z_score_rejects_invalid_arguments(self, x, mu, sigma):
        """sigma = 0 returned infinity and a negative sigma flipped the sign."""
        assert math.isnan(core.z_score(x, mu, sigma))

    def test_z_score_still_correct_when_valid(self):
        assert core.z_score(3.0, 1.0, 2.0) == pytest.approx(1.0, rel=1e-15)


# ---------------------------------------------------------------------------
# Internal consistency
#
# The same parameter was accepted by one function of a distribution and
# rejected by another.
# ---------------------------------------------------------------------------

class TestParameterRangeConsistency:
    def test_negative_binomial_accepts_p_equal_one_everywhere(self):
        """
        mean and variance accepted p = 1 and returned 0, while the PMF and CDF
        returned NaN for it. At p = 1 every trial succeeds, so the count of
        failures is 0 with certainty.
        """
        r = 5
        assert core.negative_binomial_pmf_scalar(0, r, 1.0) == 1.0
        assert core.negative_binomial_pmf_scalar(1, r, 1.0) == 0.0
        assert core.negative_binomial_cdf_scalar(0, r, 1.0) == 1.0
        assert core.negative_binomial_mean(r, 1.0) == 0.0
        assert core.negative_binomial_variance(r, 1.0) == 0.0
        assert core.negative_binomial_mgf_scalar(0.5, r, 1.0) == pytest.approx(1.0, rel=1e-15)
        assert core.negative_binomial_cgf_scalar(0.5, r, 1.0) == pytest.approx(0.0, abs=1e-15)
        assert core.negative_binomial_sample(r, 1.0) == 0

    def test_geometric_sample_defined_at_p_equal_one(self):
        """
        std::geometric_distribution is only defined for 0 < p < 1, but the
        validation admits p = 1. The first trial always succeeds there.
        """
        assert all(core.geometric_sample(1.0) == 1 for _ in range(50))

    def test_negative_binomial_still_rejects_p_above_one(self):
        for fn, args in [(core.negative_binomial_pmf_scalar, (0, 5, 1.5)),
                         (core.negative_binomial_cdf_scalar, (0, 5, 1.5)),
                         (core.negative_binomial_mean, (5, 1.5)),
                         (core.negative_binomial_variance, (5, 1.5))]:
            assert math.isnan(fn(*args)), f"{fn.__name__} accepted p = 1.5"


# ---------------------------------------------------------------------------
# The same guarantees through the Python classes
# ---------------------------------------------------------------------------

class TestThroughPublicApi:
    def test_geometric_class_cdf_small_p(self):
        got = Geometric(1e-12).cdf_scalar(3)
        assert got == pytest.approx(-math.expm1(3 * math.log1p(-1e-12)), rel=1e-15)

    def test_discrete_uniform_class_variance_full_range(self):
        assert DiscreteUniform(INT_MIN, INT_MAX).variance() > 0.0

    def test_beta_class_rejects_bad_shapes_before_reaching_the_core(self):
        with pytest.raises(ValueError):
            Beta(-1.0, -2.0)
