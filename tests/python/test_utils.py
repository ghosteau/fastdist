"""Numeric tests for the Utils statistical helpers against closed-form references."""

import math

import numpy as np
import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.utils import Utils


# ---------------------------------------------------------------------------
# Chebyshev bound
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variance, k", [(4.0, 2.0), (1.0, 1.0), (9.0, 3.0), (0.25, 0.5)])
def test_chebyshev_bound_matches_closed_form(variance, k):
    """P(|X - mu| >= k) <= sigma^2 / k^2"""
    assert Utils.chebyshev_bound(variance, k) == pytest.approx(
        variance / k ** 2, **EXACT
    )


def test_chebyshev_bound_decreases_with_k():
    values = [Utils.chebyshev_bound(4.0, k) for k in (1.0, 2.0, 4.0, 8.0)]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Bayes' rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_b_given_a, p_a, p_b", [
    (0.8, 0.25, 0.4),
    (0.9, 0.1, 0.2),
    (0.5, 0.5, 0.5),
])
def test_bayes_rule_matches_closed_form(p_b_given_a, p_a, p_b):
    """P(A|B) = P(B|A) P(A) / P(B)"""
    assert Utils.bayes_rule(p_b_given_a, p_a, p_b) == pytest.approx(
        p_b_given_a * p_a / p_b, **EXACT
    )


def test_bayes_rule_is_identity_when_evidence_matches_prior():
    """If P(B|A) == P(B), then A and B are independent and P(A|B) == P(A)."""
    assert Utils.bayes_rule(0.4, 0.25, 0.4) == pytest.approx(0.25, **EXACT)


# ---------------------------------------------------------------------------
# Law of total probability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_a, p_b_given_a", [
    ([0.3, 0.7], [0.2, 0.5]),
    ([0.5, 0.5], [1.0, 0.0]),
    ([0.2, 0.3, 0.5], [0.1, 0.4, 0.9]),
])
def test_law_of_total_probability_matches_closed_form(p_a, p_b_given_a):
    """P(B) = sum_i P(A_i) P(B|A_i)"""
    expected = sum(a * b for a, b in zip(p_a, p_b_given_a))
    assert Utils.law_of_total_probability(p_a, p_b_given_a) == pytest.approx(
        expected, **EXACT
    )


def test_law_of_total_probability_over_a_partition_is_bounded():
    result = Utils.law_of_total_probability([0.25, 0.25, 0.5], [0.9, 0.1, 0.4])
    assert 0.0 <= result <= 1.0


# REGRESSION: the signature annotates both arguments as
# Union[Real, Sequence[Real]] but they were forwarded straight to a vector-only
# binding, so scalar inputs raised TypeError from pybind11. A scalar is the
# one-element partition P(B) = P(B|A) P(A), so it is promoted rather than
# rejected and the signature now tells the truth.
def test_law_of_total_probability_accepts_scalars():
    assert Utils.law_of_total_probability(0.3, 0.2) == pytest.approx(0.06, **EXACT)


# ---------------------------------------------------------------------------
# Sigmoid / logit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 2.0, 5.0])
def test_sigmoid_matches_closed_form(x):
    assert Utils.sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)), **EXACT)


def test_sigmoid_at_zero_is_one_half():
    assert Utils.sigmoid(0.0) == pytest.approx(0.5, **EXACT)


@pytest.mark.parametrize("x", [-8.0, -2.0, 0.0, 2.0, 8.0])
def test_sigmoid_is_symmetric(x):
    """sigmoid(-x) == 1 - sigmoid(x)"""
    assert Utils.sigmoid(-x) == pytest.approx(1.0 - Utils.sigmoid(x), **EXACT)


@pytest.mark.parametrize("x", [-10.0, -1.0, 0.0, 1.0, 10.0])
def test_sigmoid_is_bounded(x):
    assert 0.0 < Utils.sigmoid(x) < 1.0


@pytest.mark.parametrize("p", [0.01, 0.25, 0.5, 0.75, 0.99])
def test_logit_matches_closed_form(p):
    assert Utils.logit(p) == pytest.approx(math.log(p / (1 - p)), **EXACT)


def test_logit_at_one_half_is_zero():
    assert Utils.logit(0.5) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("p", [0.05, 0.3, 0.5, 0.8, 0.95])
def test_logit_inverts_sigmoid(p):
    """sigmoid(logit(p)) == p"""
    assert Utils.sigmoid(Utils.logit(p)) == pytest.approx(p, **ITERATIVE)


@pytest.mark.parametrize("x", [-3.0, -0.5, 0.0, 0.5, 3.0])
def test_sigmoid_inverts_logit(x):
    """logit(sigmoid(x)) == x"""
    assert Utils.logit(Utils.sigmoid(x)) == pytest.approx(x, **ITERATIVE)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.5, 1.5])
def test_logit_returns_nan_outside_the_open_unit_interval(p):
    """
    The backend signals a domain error by returning NaN rather than raising.
    This pins the current contract; if it is changed to raise, update this test.
    """
    assert math.isnan(Utils.logit(p))


# RESOLVED: sigmoid was annotated Union[Real, Sequence[Real]] while its body
# called float() on the input, so a sequence raised an opaque numpy error. The
# annotation was the wrong half: the library's convention is an explicit *_cpu
# entry point for arrays, and returning an ndarray from a function annotated
# -> float would be worse than not accepting one. sigmoid is now scalar-only
# and says so, pointing at sigmoid_cpu.
def test_sigmoid_rejects_a_sequence_and_names_the_array_path():
    with pytest.raises(TypeError, match="sigmoid_cpu"):
        Utils.sigmoid([0.0, 2.0])


def test_sigmoid_cpu_handles_what_sigmoid_rejects():
    result = Utils.sigmoid_cpu([0.0, 2.0])
    np.testing.assert_allclose(result, [0.5, 1.0 / (1.0 + math.exp(-2.0))], rtol=1e-12)


# ---------------------------------------------------------------------------
# Batch sigmoid / logit
# ---------------------------------------------------------------------------

def test_sigmoid_cpu_matches_scalar_evaluation():
    xs = [-5.0, -1.0, 0.0, 1.0, 5.0]
    result = Utils.sigmoid_cpu(xs)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [Utils.sigmoid(x) for x in xs], rtol=1e-12)


def test_logit_cpu_matches_scalar_evaluation():
    ps = [0.1, 0.25, 0.5, 0.75, 0.9]
    result = Utils.logit_cpu(ps)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [Utils.logit(p) for p in ps], rtol=1e-12)


def test_sigmoid_cpu_accepts_numpy_input():
    xs = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        Utils.sigmoid_cpu(xs), [Utils.sigmoid(float(x)) for x in xs], rtol=1e-12
    )


def test_is_cuda_available_returns_bool():
    assert isinstance(Utils.is_cuda_available(), bool)


# ---------------------------------------------------------------------------
# Distances and similarity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x, y, expected", [
    ([0, 0], [3, 4], 5.0),
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([0, 0, 0], [1, 1, 1], math.sqrt(3)),
    ([-1, -1], [2, 3], 5.0),
])
def test_euclidean_distance_matches_closed_form(x, y, expected):
    assert Utils.euclidean_distance(x, y) == pytest.approx(expected, **EXACT)


@pytest.mark.parametrize("x, y, expected", [
    ([0, 0], [3, 4], 7.0),
    ([1, 2, 3], [1, 2, 3], 0.0),
    ([0, 0, 0], [1, 1, 1], 3.0),
    ([-1, -1], [2, 3], 7.0),
])
def test_manhattan_distance_matches_closed_form(x, y, expected):
    assert Utils.manhattan_distance(x, y) == pytest.approx(expected, **EXACT)


@pytest.mark.parametrize("x, y", [
    ([1, 2, 3], [4, 5, 6]),
    ([0, 1], [1, 0]),
    ([-3, 7, 2], [5, -1, 4]),
])
def test_euclidean_distance_matches_numpy(x, y):
    assert Utils.euclidean_distance(x, y) == pytest.approx(
        float(np.linalg.norm(np.array(x, float) - np.array(y, float))), **EXACT
    )


@pytest.mark.parametrize("x, y", [([1, 2, 3], [4, 5, 6]), ([-3, 7, 2], [5, -1, 4])])
def test_distances_are_symmetric(x, y):
    assert Utils.euclidean_distance(x, y) == pytest.approx(
        Utils.euclidean_distance(y, x), **EXACT
    )
    assert Utils.manhattan_distance(x, y) == pytest.approx(
        Utils.manhattan_distance(y, x), **EXACT
    )


@pytest.mark.parametrize("x, y", [([1, 2, 3], [4, 5, 6]), ([0, 1], [1, 0])])
def test_euclidean_never_exceeds_manhattan(x, y):
    """The L2 norm is bounded above by the L1 norm."""
    assert Utils.euclidean_distance(x, y) <= Utils.manhattan_distance(x, y) + 1e-12


@pytest.mark.parametrize("x, y, expected", [
    ([1, 0], [0, 1], 0.0),
    ([1, 2, 3], [1, 2, 3], 1.0),
    ([1, 0], [-1, 0], -1.0),
    ([1, 1], [2, 2], 1.0),
])
def test_cosine_similarity_matches_closed_form(x, y, expected):
    assert Utils.cosine_similarity(x, y) == pytest.approx(expected, **ITERATIVE)


@pytest.mark.parametrize("x, y", [([1, 2, 3], [4, 5, 6]), ([-3, 7, 2], [5, -1, 4])])
def test_cosine_similarity_matches_numpy(x, y):
    xv, yv = np.array(x, float), np.array(y, float)
    expected = float(xv @ yv / (np.linalg.norm(xv) * np.linalg.norm(yv)))
    assert Utils.cosine_similarity(x, y) == pytest.approx(expected, **ITERATIVE)


@pytest.mark.parametrize("x, y", [([1, 2, 3], [4, 5, 6]), ([0, 1], [1, 0])])
def test_cosine_similarity_is_bounded(x, y):
    assert -1.0 <= Utils.cosine_similarity(x, y) <= 1.0


def test_cosine_similarity_is_scale_invariant():
    base = Utils.cosine_similarity([1, 2, 3], [4, 5, 6])
    scaled = Utils.cosine_similarity([10, 20, 30], [4, 5, 6])
    assert base == pytest.approx(scaled, **ITERATIVE)


def test_cosine_similarity_of_a_zero_vector_is_nan():
    """The zero vector has no direction; the backend signals this with NaN."""
    assert math.isnan(Utils.cosine_similarity([0, 0], [1, 1]))


@pytest.mark.parametrize("fn", [
    Utils.euclidean_distance,
    Utils.manhattan_distance,
    Utils.cosine_similarity,
])
def test_distance_functions_reject_mismatched_lengths(fn):
    with pytest.raises(ValueError, match="x and y must have the same length"):
        fn([1, 2], [1, 2, 3])


@pytest.mark.parametrize("fn", [
    Utils.euclidean_distance,
    Utils.manhattan_distance,
    Utils.cosine_similarity,
])
def test_distance_functions_accept_numpy_input(fn):
    assert fn(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(
        fn([1.0, 2.0], [3.0, 4.0]), **EXACT
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mean, stddev", [(10.0, 2.0), (5.0, 5.0), (100.0, 1.0)])
def test_coefficient_of_variation_matches_closed_form(mean, stddev):
    """CV = sigma / mu"""
    assert Utils.coefficient_of_variation(mean, stddev) == pytest.approx(
        stddev / mean, **EXACT
    )


@pytest.mark.parametrize("mean_x, mean_y, e_xy", [
    (2.0, 3.0, 7.0),
    (0.0, 0.0, 1.0),
    (-1.0, 4.0, 2.0),
])
def test_covariance_matches_closed_form(mean_x, mean_y, e_xy):
    """Cov(X, Y) = E[XY] - E[X]E[Y]"""
    assert Utils.covariance(mean_x, mean_y, e_xy) == pytest.approx(
        e_xy - mean_x * mean_y, **EXACT
    )


def test_covariance_is_zero_for_independent_variables():
    """If E[XY] == E[X]E[Y] the covariance vanishes."""
    assert Utils.covariance(2.0, 3.0, 6.0) == pytest.approx(0.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, k", [(10, 3), (5, 0), (5, 5), (52, 5), (100, 50)])
def test_choose_matches_reference(n, k):
    assert Utils.choose(n, k) == pytest.approx(float(math.comb(n, k)), rel=1e-12)


@pytest.mark.parametrize("n, k", [(5, 7), (3, 10)])
def test_choose_is_zero_when_k_exceeds_n(n, k):
    assert Utils.choose(n, k) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("n, k", [(10, 3), (52, 5), (20, 10)])
def test_choose_is_symmetric(n, k):
    """C(n, k) == C(n, n-k)"""
    assert Utils.choose(n, k) == pytest.approx(Utils.choose(n, n - k), rel=1e-12)


@pytest.mark.parametrize("n, k", [(10, 3), (5, 0), (5, 5), (20, 4)])
def test_permutation_matches_reference(n, k):
    assert Utils.permutation(n, k) == pytest.approx(
        float(math.perm(n, k)), rel=1e-12
    )


@pytest.mark.parametrize("n, k", [(10, 3), (20, 4), (52, 5)])
def test_permutation_equals_choose_times_factorial(n, k):
    """P(n, k) == C(n, k) * k!"""
    assert Utils.permutation(n, k) == pytest.approx(
        Utils.choose(n, k) * math.factorial(k), rel=1e-10
    )


@pytest.mark.parametrize("n", [0, 1, 5, 10, 20, 100, 170])
def test_factorial_matches_reference(n):
    assert Utils.factorial(n) == pytest.approx(float(math.factorial(n)), rel=1e-12)


def test_factorial_overflows_beyond_the_double_range():
    """171! exceeds the maximum finite double, so inf is the correct result."""
    assert math.isinf(Utils.factorial(171))


@pytest.mark.parametrize("n, a, b", [(3, 1.0, 2.0), (5, 2.0, 3.0), (0, 4.0, 7.0)])
def test_binomial_theorem_matches_closed_form(n, a, b):
    """The binomial theorem expands to (a + b)^n."""
    assert Utils.binomial(n, a, b) == pytest.approx((a + b) ** n, rel=1e-10)


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 5.0, 7.5, 10.0])
def test_gamma_matches_reference(x):
    assert Utils.gamma(x) == pytest.approx(math.gamma(x), rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_gamma_of_an_integer_is_a_factorial(n):
    """Gamma(n) == (n-1)!"""
    assert Utils.gamma(float(n)) == pytest.approx(
        float(math.factorial(n - 1)), rel=1e-10
    )


def test_gamma_of_one_half_is_root_pi():
    assert Utils.gamma(0.5) == pytest.approx(math.sqrt(math.pi), rel=1e-12)


@pytest.mark.parametrize("x", [0.5, 1.0, 2.5, 10.0, 100.0])
def test_log_gamma_matches_reference(x):
    assert Utils.log_gamma(x) == pytest.approx(math.lgamma(x), rel=1e-12)


@pytest.mark.parametrize("x", [0.5, 1.5, 4.0, 9.0])
def test_log_gamma_is_log_of_gamma(x):
    assert Utils.log_gamma(x) == pytest.approx(math.log(Utils.gamma(x)), **ITERATIVE)


@pytest.mark.parametrize("x", [1.5, 3.0, 7.25])
def test_gamma_recurrence(x):
    """Gamma(x + 1) == x * Gamma(x)"""
    assert Utils.gamma(x + 1.0) == pytest.approx(x * Utils.gamma(x), rel=1e-10)
