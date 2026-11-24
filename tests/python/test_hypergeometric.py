"""
Numeric tests for the Hypergeometric distribution.

The reference is exact rational arithmetic wherever it is affordable: the PMF
is a ratio of binomial coefficients, so Python's integer ``math.comb`` and
``Fraction`` give the true value to unlimited precision, and the backend is
measured against that rather than against SciPy or a recorded number.

    population  N  size of the whole population
    successes   K  items in the population that count as a success
    draws       n  items drawn, without replacement
    k              successes observed among the draws

    P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)
"""

import math
from collections import Counter
from fractions import Fraction

import pytest

from conftest import EXACT
from probforge import _probforge as core
from probforge.distributions.hypergeometric import Hypergeometric

# Accuracy the log-space PMF can actually deliver.
#
# The PMF is evaluated as a sum of lgamma terms, and that approach has a hard
# floor: at a population of 1000, lgamma(1001) is about 5912.13, whose ULP is
# already 9.1e-13. The log simply cannot be represented more precisely than
# that, so the exponentiated result carries a relative error of the same order
# however carefully the sum is arranged. Measured worst case over the grid
# below is 1.4e-12; SciPy reaches 8e-16 by never forming a large log at all
# (Loader, "Fast and Accurate Computation of Binomial Probabilities", 2000).
#
# These tests hold the backend to what its method can do, and would catch a
# regression of more than a few multiples of it.
LOG_SPACE = {"rel": 1e-11, "abs": 1e-15}


# ---------------------------------------------------------------------------
# Exact references
# ---------------------------------------------------------------------------

def exact_pmf(k: int, N: int, K: int, n: int) -> Fraction:
    """P(X = k) in exact rational arithmetic."""
    lo = max(0, n - (N - K))
    hi = min(n, K)
    if k < lo or k > hi:
        return Fraction(0)
    return Fraction(math.comb(K, k) * math.comb(N - K, n - k), math.comb(N, n))


def exact_cdf(k: int, N: int, K: int, n: int) -> Fraction:
    lo = max(0, n - (N - K))
    return sum((exact_pmf(j, N, K, n) for j in range(lo, k + 1)), Fraction(0))


def exact_support(N: int, K: int, n: int):
    return max(0, n - (N - K)), min(n, K)


# A spread of shapes: symmetric, skewed, degenerate, draw-everything, and
# cases where the support does not start at zero.
PARAM_GRID = [
    (20, 7, 5),
    (50, 10, 10),
    (10, 5, 5),
    (10, 9, 9),      # support starts above zero: lo = 8
    (100, 50, 30),
    (6, 3, 3),
    (2, 1, 1),
    (1000, 2, 500),
    (1000, 998, 999),  # lo = 997
    (15, 15, 7),     # every item is a success
    (15, 0, 7),      # no successes at all
    (12, 4, 0),      # an empty draw
    (12, 4, 12),     # the whole population drawn
]


# ---------------------------------------------------------------------------
# PMF and CDF against exact rational values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_pmf_matches_exact_rational(N, K, n):
    """P(X = k) against exact rational arithmetic, across and just outside the support."""
    lo, hi = exact_support(N, K, n)
    for k in range(max(0, lo - 2), hi + 3):
        want = exact_pmf(k, N, K, n)
        got = core.hypergeometric_pmf_scalar(k=k, population=N, successes=K, draws=n)
        if want == 0:
            assert got == 0.0, f"k={k} is outside the support but the PMF is {got}"
        else:
            assert got == pytest.approx(float(want), **LOG_SPACE)


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_cdf_matches_exact_rational(N, K, n):
    """P(X <= k) against the exact rational partial sums of the PMF."""
    lo, hi = exact_support(N, K, n)
    for k in range(max(0, lo - 2), hi + 3):
        want = float(exact_cdf(k, N, K, n))
        got = core.hypergeometric_cdf_scalar(k=k, population=N, successes=K, draws=n)
        assert got == pytest.approx(want, **LOG_SPACE)


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_pmf_sums_to_one_over_the_support(N, K, n):
    """The mass over the support is all of it -- Vandermonde's identity in disguise."""
    lo, hi = exact_support(N, K, n)
    total = sum(core.hypergeometric_pmf_scalar(k=k, population=N, successes=K, draws=n)
                for k in range(lo, hi + 1))
    assert total == pytest.approx(1.0, **LOG_SPACE)


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_cdf_is_one_at_the_top_of_the_support(N, K, n):
    """The CDF is pinned to 1 at the top of the support and stays there above it."""
    _, hi = exact_support(N, K, n)
    assert core.hypergeometric_cdf_scalar(k=hi, population=N, successes=K, draws=n) == 1.0
    assert core.hypergeometric_cdf_scalar(k=hi + 50, population=N, successes=K, draws=n) == 1.0


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_cdf_is_zero_below_the_support(N, K, n):
    """The CDF is 0 below the support, which does not always start at zero."""
    lo, _ = exact_support(N, K, n)
    assert core.hypergeometric_cdf_scalar(k=lo - 1, population=N, successes=K, draws=n) == 0.0
    assert core.hypergeometric_cdf_scalar(k=-100, population=N, successes=K, draws=n) == 0.0


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_cdf_is_monotone(N, K, n):
    """A CDF must never decrease and never leave [0, 1], including at the edges."""
    lo, hi = exact_support(N, K, n)
    previous = -1.0
    for k in range(lo - 1, hi + 2):
        current = core.hypergeometric_cdf_scalar(k=k, population=N, successes=K, draws=n)
        assert current >= previous - 1e-15, f"CDF decreased at k={k}"
        assert 0.0 <= current <= 1.0, f"CDF left [0, 1] at k={k}: {current}"
        previous = current


# ---------------------------------------------------------------------------
# log PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_logpmf_is_the_log_of_the_pmf(N, K, n):
    """The two routes must agree wherever the PMF is representable."""
    lo, hi = exact_support(N, K, n)
    for k in range(lo, hi + 1):
        pmf = core.hypergeometric_pmf_scalar(k=k, population=N, successes=K, draws=n)
        logpmf = core.hypergeometric_logpmf_scalar(k=k, population=N, successes=K, draws=n)
        assert logpmf == pytest.approx(math.log(pmf), rel=1e-12, abs=1e-12)


def test_logpmf_outside_the_support_is_negative_infinity():
    """Zero probability has log -inf, not NaN, so sums over the support stay usable."""
    assert core.hypergeometric_logpmf_scalar(k=-1, population=20, successes=7, draws=5) == -math.inf
    assert core.hypergeometric_logpmf_scalar(k=6, population=20, successes=7, draws=5) == -math.inf


def test_logpmf_survives_a_population_too_large_for_direct_factorials():
    """
    C(1_000_000, 500_000) overflows a double many times over, but the PMF it
    appears in is an ordinary number. Evaluating in log space is what keeps
    this finite.
    """
    value = core.hypergeometric_logpmf_scalar(k=250, population=1_000_000, successes=500_000, draws=500)
    assert math.isfinite(value)
    pmf = core.hypergeometric_pmf_scalar(k=250, population=1_000_000, successes=500_000, draws=500)
    assert 0.0 < pmf < 1.0
    # Drawing 500 from a half-success population is near-binomial(500, 1/2).
    assert pmf == pytest.approx(math.comb(500, 250) / 2.0**500, rel=1e-3)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_mean_matches_the_exact_sum(N, K, n):
    """The closed-form mean against sum_k k P(k), evaluated in exact rationals."""
    lo, hi = exact_support(N, K, n)
    want = float(sum(k * exact_pmf(k, N, K, n) for k in range(lo, hi + 1)))
    got = core.hypergeometric_mean(population=N, successes=K, draws=n)
    assert got == pytest.approx(want, **EXACT)


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_variance_matches_the_exact_sum(N, K, n):
    """The closed form against sum_k (k - mean)^2 P(k), evaluated in exact rationals."""
    lo, hi = exact_support(N, K, n)
    mean = sum(k * exact_pmf(k, N, K, n) for k in range(lo, hi + 1))
    want = float(sum((k - mean) ** 2 * exact_pmf(k, N, K, n) for k in range(lo, hi + 1)))
    got = core.hypergeometric_variance(population=N, successes=K, draws=n)
    assert got == pytest.approx(want, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("N, K, n", PARAM_GRID)
def test_stddev_is_the_root_of_the_variance(N, K, n):
    """The two must stay consistent; stddev delegates to variance."""
    var = core.hypergeometric_variance(population=N, successes=K, draws=n)
    got = core.hypergeometric_stddev(population=N, successes=K, draws=n)
    assert got == pytest.approx(math.sqrt(var), **EXACT)


def test_variance_is_zero_when_the_whole_population_is_drawn():
    """Drawing everything leaves nothing to vary."""
    assert core.hypergeometric_variance(population=12, successes=4, draws=12) == 0.0


def test_variance_of_a_single_item_population_is_zero():
    """
    The finite population correction is (N-n)/(N-1), which is 0/0 at N = 1.
    The distribution is deterministic there, so the variance is zero. SciPy
    reports nan for this case.
    """
    assert core.hypergeometric_variance(population=1, successes=1, draws=1) == 0.0
    assert core.hypergeometric_variance(population=1, successes=0, draws=1) == 0.0


def test_variance_is_below_the_binomial_variance():
    """
    Sampling without replacement is the only difference from a binomial, and
    it can only reduce the spread.
    """
    N, K, n = 100, 40, 20
    p = K / N
    binomial_var = n * p * (1 - p)
    assert core.hypergeometric_variance(population=N, successes=K, draws=n) < binomial_var


# ---------------------------------------------------------------------------
# MGF and CGF
#
# The hypergeometric MGF has no elementary closed form, so it is accumulated
# over the finite support. The reference here is that same defining sum built
# from exact rational probabilities.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N, K, n", [(20, 7, 5), (50, 10, 10), (10, 9, 9), (15, 0, 7)])
@pytest.mark.parametrize("t", [0.0, 0.3, -0.7, 1.5, -2.0])
def test_mgf_matches_the_defining_sum(N, K, n, t):
    """The MGF against sum_k exp(tk) P(k) built from exact rational probabilities."""
    lo, hi = exact_support(N, K, n)
    want = sum(math.exp(t * k) * float(exact_pmf(k, N, K, n)) for k in range(lo, hi + 1))
    got = core.hypergeometric_mgf_scalar(t=t, population=N, successes=K, draws=n)
    assert got == pytest.approx(want, **LOG_SPACE)


def test_mgf_at_zero_is_one():
    """M(0) = 1 for every shape, since it reduces to the total mass."""
    for N, K, n in PARAM_GRID:
        got = core.hypergeometric_mgf_scalar(t=0.0, population=N, successes=K, draws=n)
        assert got == pytest.approx(1.0, **LOG_SPACE)


@pytest.mark.parametrize("N, K, n", [(20, 7, 5), (50, 10, 10)])
@pytest.mark.parametrize("t", [0.0, 0.3, -0.7])
def test_cgf_is_the_log_of_the_mgf(N, K, n, t):
    """The CGF is log M(t) wherever the MGF itself is representable."""
    mgf = core.hypergeometric_mgf_scalar(t=t, population=N, successes=K, draws=n)
    cgf = core.hypergeometric_cgf_scalar(t=t, population=N, successes=K, draws=n)
    assert cgf == pytest.approx(math.log(mgf), rel=1e-12, abs=1e-15)


def test_cgf_survives_a_t_that_overflows_the_mgf():
    """
    log-sum-exp factors the largest exponent out before exponentiating, so the
    CGF stays finite where the MGF itself cannot be represented. exp(200000)
    is far outside a double; the CGF of the same point is about 199709.
    """
    N, K, n, t = 1000, 400, 600, 500.0
    cgf = core.hypergeometric_cgf_scalar(t=t, population=N, successes=K, draws=n)
    assert math.isfinite(cgf)
    assert math.isinf(core.hypergeometric_mgf_scalar(t=t, population=N, successes=K, draws=n))

    # At a t this large the sum is dominated by the top of the support, whose
    # term is exp(t*k_max) * P(k_max) -- the log-probability there is -290.8,
    # not negligible, so the CGF sits just below t * k_max.
    top = min(n, K)
    log_p_top = core.hypergeometric_logpmf_scalar(k=top, population=N, successes=K, draws=n)
    assert cgf == pytest.approx(t * top + log_p_top, rel=1e-12)
    assert cgf < t * top


def test_mgf_derivative_at_zero_recovers_the_mean():
    """M'(0) = E[X], checked with a central difference."""
    N, K, n = 50, 20, 15
    h = 1e-5
    up = core.hypergeometric_mgf_scalar(t=h, population=N, successes=K, draws=n)
    down = core.hypergeometric_mgf_scalar(t=-h, population=N, successes=K, draws=n)
    derivative = (up - down) / (2 * h)
    assert derivative == pytest.approx(core.hypergeometric_mean(population=N, successes=K, draws=n), rel=1e-7)


def test_cgf_second_derivative_at_zero_recovers_the_variance():
    """K''(0) = Var[X], checked with a central second difference."""
    N, K, n = 50, 20, 15
    h = 1e-3
    up = core.hypergeometric_cgf_scalar(t=h, population=N, successes=K, draws=n)
    mid = core.hypergeometric_cgf_scalar(t=0.0, population=N, successes=K, draws=n)
    down = core.hypergeometric_cgf_scalar(t=-h, population=N, successes=K, draws=n)
    second = (up - 2 * mid + down) / (h * h)
    assert second == pytest.approx(core.hypergeometric_variance(population=N, successes=K, draws=n), rel=1e-5)


# ---------------------------------------------------------------------------
# Identities that tie the distribution to its neighbours
# ---------------------------------------------------------------------------

def test_symmetry_in_successes_and_draws():
    """
    Choosing which n items to draw and which K items are successes is the same
    experiment viewed two ways, so P(k; N, K, n) = P(k; N, n, K).
    """
    for N, K, n in [(20, 7, 5), (50, 10, 30), (100, 60, 40)]:
        for k in range(0, min(K, n) + 1):
            a = core.hypergeometric_pmf_scalar(k=k, population=N, successes=K, draws=n)
            b = core.hypergeometric_pmf_scalar(k=k, population=N, successes=n, draws=K)
            assert a == pytest.approx(b, rel=1e-12)


def test_approaches_the_binomial_for_a_large_population():
    """
    With the success proportion held fixed, sampling without replacement from
    a huge pool is indistinguishable from sampling with replacement.
    """
    n, p = 10, 0.25
    N = 20_000_000
    K = int(N * p)
    for k in range(n + 1):
        hyper = core.hypergeometric_pmf_scalar(k=k, population=N, successes=K, draws=n)
        binom = math.comb(n, k) * p**k * (1 - p) ** (n - k)
        assert hyper == pytest.approx(binom, rel=1e-5)


def test_single_draw_is_bernoulli():
    """One draw succeeds with probability K/N."""
    N, K = 37, 11
    assert core.hypergeometric_pmf_scalar(k=1, population=N, successes=K, draws=1) == pytest.approx(K / N, **EXACT)
    assert core.hypergeometric_pmf_scalar(k=0, population=N, successes=K, draws=1) == pytest.approx(
        (N - K) / N, **EXACT)


def test_degenerate_cases():
    """An empty draw, an all-success pool and a no-success pool are certain."""
    assert core.hypergeometric_pmf_scalar(k=0, population=12, successes=4, draws=0) == 1.0
    assert core.hypergeometric_pmf_scalar(k=7, population=15, successes=15, draws=7) == 1.0
    assert core.hypergeometric_pmf_scalar(k=0, population=15, successes=0, draws=7) == 1.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N, K, n", [
    (-1, 0, 0),      # negative population
    (10, -1, 5),     # negative successes
    (10, 5, -1),     # negative draws
    (10, 11, 5),     # more successes than the population holds
    (10, 5, 11),     # more draws than the population holds
])
def test_core_rejects_impossible_parameters(N, K, n):
    """Parameters that cannot describe a draw yield NaN, and -1 from the sampler."""
    assert math.isnan(core.hypergeometric_pmf_scalar(k=0, population=N, successes=K, draws=n))
    assert math.isnan(core.hypergeometric_cdf_scalar(k=0, population=N, successes=K, draws=n))
    assert math.isnan(core.hypergeometric_mean(population=N, successes=K, draws=n))
    assert math.isnan(core.hypergeometric_variance(population=N, successes=K, draws=n))
    assert core.hypergeometric_sample(population=N, successes=K, draws=n) == -1


def test_core_rejects_an_empty_population_for_moments():
    """With no population there is no distribution to take a mean of."""
    assert math.isnan(core.hypergeometric_mean(population=0, successes=0, draws=0))
    assert math.isnan(core.hypergeometric_variance(population=0, successes=0, draws=0))


@pytest.mark.parametrize("kwargs", [
    {"population": -1, "successes": 0, "draws": 0},
    {"population": 10, "successes": -1, "draws": 5},
    {"population": 10, "successes": 5, "draws": -1},
    {"population": 10, "successes": 11, "draws": 5},
    {"population": 10, "successes": 5, "draws": 11},
])
def test_class_refuses_impossible_parameters(kwargs):
    """The class refuses at construction what the core would answer with NaN."""
    with pytest.raises(ValueError):
        Hypergeometric(**kwargs)


@pytest.mark.parametrize("bad", [2.5, "5", None, [5], True])
def test_class_refuses_non_integer_parameters(bad):
    """Counts must be integers; bool is refused despite subclassing int."""
    with pytest.raises(TypeError):
        Hypergeometric(population=bad, successes=1, draws=1)


@pytest.mark.parametrize("bad", [2.5, "2", None, [2], True])
def test_class_refuses_non_integer_k(bad):
    """k is a count, so a float is a mistake rather than something to round."""
    with pytest.raises(TypeError):
        Hypergeometric(population=20, successes=7, draws=5).pmf_scalar(bad)


def test_setters_revalidate_against_the_other_parameters():
    """
    The constraints are between the parameters, so a setter cannot check its
    own value alone.
    """
    h = Hypergeometric(population=20, successes=7, draws=5)
    with pytest.raises(ValueError):
        h.successes = 25            # more successes than the population holds
    with pytest.raises(ValueError):
        h.draws = 25                # more draws than the population holds
    with pytest.raises(ValueError):
        h.population = 3            # now too small for successes = 7
    # The instance is unchanged after every rejected assignment.
    assert (h.population, h.successes, h.draws) == (20, 7, 5)


def test_setters_accept_a_consistent_change():
    """A change that keeps the triple consistent is applied and takes effect."""
    h = Hypergeometric(population=20, successes=7, draws=5)
    h.population = 30
    assert h.population == 30
    h.successes = 25
    assert h.successes == 25
    assert h.mean() == pytest.approx(5 * 25 / 30, **EXACT)


# ---------------------------------------------------------------------------
# Sampling
#
# There is no std::hypergeometric_distribution, so the sampler is written by
# hand and needs to be held to the distribution it claims to draw from.
# ---------------------------------------------------------------------------

def test_sample_stays_inside_the_support():
    """Every draw lands in the support, including shapes whose support starts above zero."""
    core.seed(20240101)
    for N, K, n in PARAM_GRID:
        lo, hi = exact_support(N, K, n)
        for _ in range(200):
            draw = core.hypergeometric_sample(population=N, successes=K, draws=n)
            assert lo <= draw <= hi, f"draw {draw} outside [{lo}, {hi}] for N={N} K={K} n={n}"


def test_sample_is_reproducible_under_a_seed():
    """The same seed replays the same draws, which is what makes these tests deterministic."""
    core.seed(777)
    first = [core.hypergeometric_sample(population=50, successes=20, draws=12) for _ in range(100)]
    core.seed(777)
    second = [core.hypergeometric_sample(population=50, successes=20, draws=12) for _ in range(100)]
    assert first == second


def test_sample_matches_the_distribution_by_chi_square():
    """
    Pearson's chi-square against the exact PMF. The seed is fixed, so this is
    a deterministic check on the sampler, not a flaky statistical one; the
    threshold is the 99.9th percentile of the null, so a correct sampler has
    about one chance in a thousand of tripping it at any given seed.
    """
    N, K, n = 50, 20, 12
    trials = 200_000
    core.seed(31337)

    counts = Counter(core.hypergeometric_sample(population=N, successes=K, draws=n)
                     for _ in range(trials))

    lo, hi = exact_support(N, K, n)
    statistic = 0.0
    degrees = -1
    for k in range(lo, hi + 1):
        expected = float(exact_pmf(k, N, K, n)) * trials
        if expected < 5.0:          # pool away the cells too thin to test
            continue
        statistic += (counts.get(k, 0) - expected) ** 2 / expected
        degrees += 1

    # chi-square 99.9th percentile for the degrees of freedom involved here
    # (9 or 10) is about 29.6; take a round number safely above it.
    assert degrees >= 5, "the grid should leave several testable cells"
    assert statistic < 32.0, f"chi-square {statistic:.2f} on {degrees} df: the sampler does not match the PMF"


def test_sample_mean_converges_to_the_analytic_mean():
    """The sample mean lands within five standard errors of the analytic mean."""
    N, K, n = 100, 30, 25
    trials = 100_000
    core.seed(4242)
    total = sum(core.hypergeometric_sample(population=N, successes=K, draws=n) for _ in range(trials))
    observed = total / trials
    expected = core.hypergeometric_mean(population=N, successes=K, draws=n)
    standard_error = math.sqrt(core.hypergeometric_variance(population=N, successes=K, draws=n) / trials)
    assert abs(observed - expected) < 5 * standard_error


def test_sample_variance_converges_to_the_analytic_variance():
    """The sample variance matches the analytic variance to within 5%."""
    N, K, n = 100, 30, 25
    trials = 100_000
    core.seed(99)
    draws = [core.hypergeometric_sample(population=N, successes=K, draws=n) for _ in range(trials)]
    mean = sum(draws) / trials
    observed = sum((d - mean) ** 2 for d in draws) / (trials - 1)
    expected = core.hypergeometric_variance(population=N, successes=K, draws=n)
    assert observed == pytest.approx(expected, rel=0.05)


def test_degenerate_sampling_is_deterministic():
    """An empty draw, an all-success pool, a no-success pool and a full draw are certain."""
    core.seed(5)
    assert all(core.hypergeometric_sample(population=12, successes=4, draws=0) == 0 for _ in range(20))
    assert all(core.hypergeometric_sample(population=15, successes=15, draws=7) == 7 for _ in range(20))
    assert all(core.hypergeometric_sample(population=15, successes=0, draws=7) == 0 for _ in range(20))
    assert all(core.hypergeometric_sample(population=12, successes=4, draws=12) == 4 for _ in range(20))


# ---------------------------------------------------------------------------
# The class wraps the core faithfully
# ---------------------------------------------------------------------------

def test_instance_methods_agree_with_the_core():
    """The class forwards to the core without altering any value."""
    N, K, n = 40, 15, 9
    h = Hypergeometric(population=N, successes=K, draws=n)
    for k in range(0, 10):
        assert h.pmf_scalar(k) == core.hypergeometric_pmf_scalar(k=k, population=N, successes=K, draws=n)
        assert h.cdf_scalar(k) == core.hypergeometric_cdf_scalar(k=k, population=N, successes=K, draws=n)
    assert h.mean() == core.hypergeometric_mean(population=N, successes=K, draws=n)
    assert h.variance() == core.hypergeometric_variance(population=N, successes=K, draws=n)
    assert h.stddev() == core.hypergeometric_stddev(population=N, successes=K, draws=n)
    assert h.mgf_scalar(0.25) == core.hypergeometric_mgf_scalar(t=0.25, population=N, successes=K, draws=n)
    assert h.cgf_scalar(0.25) == core.hypergeometric_cgf_scalar(t=0.25, population=N, successes=K, draws=n)


def test_classmethods_agree_with_instance_methods():
    """The parameter-taking classmethods match the instance methods exactly."""
    N, K, n = 40, 15, 9
    h = Hypergeometric(population=N, successes=K, draws=n)
    assert Hypergeometric._pmf_scalar(3, N, K, n) == h.pmf_scalar(3)
    assert Hypergeometric._cdf_scalar(3, N, K, n) == h.cdf_scalar(3)
    assert Hypergeometric._logpmf_scalar(3, N, K, n) == h.logpmf_scalar(3)
    assert Hypergeometric._mean(N, K, n) == h.mean()
    assert Hypergeometric._variance(N, K, n) == h.variance()
    assert Hypergeometric._stddev(N, K, n) == h.stddev()
    assert Hypergeometric._mgf_scalar(0.5, N, K, n) == h.mgf_scalar(0.5)
    assert Hypergeometric._cgf_scalar(0.5, N, K, n) == h.cgf_scalar(0.5)


def test_support_matches_the_definition():
    """support() agrees with max(0, n - (N - K)) .. min(n, K) for every shape."""
    for N, K, n in PARAM_GRID:
        assert Hypergeometric(population=N, successes=K, draws=n).support() == exact_support(N, K, n)


def test_repr_round_trips():
    """repr() is valid Python that rebuilds an equivalent distribution."""
    h = Hypergeometric(population=20, successes=7, draws=5)
    assert repr(h) == "Hypergeometric(population=20, successes=7, draws=5)"
    assert eval(repr(h)).support() == h.support()  # noqa: S307 - fixed string from above


def test_exported_from_the_package_root():
    """Hypergeometric is reachable as probforge.Hypergeometric and listed in __all__."""
    import probforge
    assert probforge.Hypergeometric is Hypergeometric
    assert "Hypergeometric" in probforge.__all__
