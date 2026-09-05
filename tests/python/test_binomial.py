"""Numeric tests for the Binomial distribution against closed-form references."""

import math

import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.binomial import Binomial


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) = C(n, k) p^k (1-p)^(n-k)"""
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def binom_cdf(k: int, n: int, p: float) -> float:
    return sum(binom_pmf(i, n, p) for i in range(0, k + 1))


def binom_mgf(t: float, n: int, p: float) -> float:
    """M(t) = (1 - p + p e^t)^n"""
    return (1 - p + p * math.exp(t)) ** n


PARAMS = [(1, 0.5), (10, 0.3), (10, 0.7), (5, 0.5), (20, 0.1)]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    dist = Binomial(n=10, p=0.3)
    assert dist.n == 10
    assert dist.p == 0.3


@pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
def test_init_rejects_out_of_range_p(p):
    with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
        Binomial(n=10, p=p)


def test_init_rejects_negative_n():
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        Binomial(n=-1, p=0.5)


@pytest.mark.parametrize("bad_n", [2.5, "10"])
def test_init_rejects_non_integer_n(bad_n):
    with pytest.raises(TypeError, match="n must be an integer"):
        Binomial(n=bad_n, p=0.5)


@pytest.mark.parametrize("bad_p", ["0.5", object()])
def test_init_rejects_non_real_p(bad_p):
    with pytest.raises(TypeError, match="p must be a real number"):
        Binomial(n=10, p=bad_p)


@pytest.mark.parametrize("kwargs", [{"n": None, "p": 0.5}, {"n": 10, "p": None}])
def test_init_rejects_none_parameters(kwargs):
    """
    None is rejected, but only incidentally. _validate_params skips its type
    check when a parameter is None, so the failure surfaces from an unguarded
    comparison or from float() rather than as the intended message.
    """
    with pytest.raises(TypeError):
        Binomial(**kwargs)


def test_repr():
    assert repr(Binomial(n=10, p=0.3)) == "Binomial(n=10, p=0.3)"


def test_n_setter_updates_and_validates():
    dist = Binomial(n=10, p=0.3)
    dist.n = 20
    assert dist.n == 20
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        dist.n = -5


# REGRESSION: the `if n < 0` check sat outside the `if n is not None` guard, so
# setting p (which leaves n as None) raised TypeError comparing None to int for
# every assignment. Same defect pattern as Beta._validate_params had.
def test_p_setter_updates_value():
    dist = Binomial(n=10, p=0.3)
    dist.p = 0.6
    assert dist.p == 0.6


def test_p_setter_rejects_out_of_range():
    dist = Binomial(n=10, p=0.3)
    with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
        dist.p = 1.5


# ---------------------------------------------------------------------------
# PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", PARAMS)
def test_pmf_matches_closed_form(n, p):
    dist = Binomial(n, p)
    for k in range(n + 1):
        assert dist.pmf_scalar(k) == pytest.approx(binom_pmf(k, n, p), **EXACT)


@pytest.mark.parametrize("n, p", PARAMS)
def test_pmf_sums_to_one_over_the_support(n, p):
    dist = Binomial(n, p)
    total = sum(dist.pmf_scalar(k) for k in range(n + 1))
    assert total == pytest.approx(1.0, **ITERATIVE)


@pytest.mark.parametrize("n, p", PARAMS)
def test_pmf_is_a_probability(n, p):
    dist = Binomial(n, p)
    for k in range(n + 1):
        assert 0.0 <= dist.pmf_scalar(k) <= 1.0


def test_pmf_is_symmetric_for_fair_coin():
    """For p = 0.5, P(X = k) == P(X = n - k)."""
    dist = Binomial(10, 0.5)
    for k in range(11):
        assert dist.pmf_scalar(k) == pytest.approx(dist.pmf_scalar(10 - k), **EXACT)


# ---------------------------------------------------------------------------
# log-PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", [(10, 0.3), (20, 0.1), (5, 0.5)])
def test_logpmf_is_log_of_pmf(n, p):
    dist = Binomial(n, p)
    for k in range(n + 1):
        assert dist.logpmf_scalar(k) == pytest.approx(
            math.log(binom_pmf(k, n, p)), **ITERATIVE
        )


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", PARAMS)
def test_cdf_matches_summed_pmf(n, p):
    dist = Binomial(n, p)
    for k in range(n + 1):
        assert dist.cdf_scalar(k) == pytest.approx(binom_cdf(k, n, p), **ITERATIVE)


@pytest.mark.parametrize("n, p", PARAMS)
def test_cdf_is_monotonic_and_bounded(n, p):
    dist = Binomial(n, p)
    values = [dist.cdf_scalar(k) for k in range(n + 1)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("n, p", PARAMS)
def test_cdf_reaches_one_at_the_top_of_the_support(n, p):
    assert Binomial(n, p).cdf_scalar(n) == pytest.approx(1.0, **ITERATIVE)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", PARAMS)
def test_moments_match_closed_forms(n, p):
    dist = Binomial(n, p)
    assert dist.mean() == pytest.approx(n * p, **EXACT)
    assert dist.variance() == pytest.approx(n * p * (1 - p), **EXACT)
    assert dist.stddev() == pytest.approx(math.sqrt(n * p * (1 - p)), **EXACT)


@pytest.mark.parametrize("n, p", PARAMS)
def test_mean_equals_pmf_weighted_sum(n, p):
    """E[X] = sum k P(X = k), computed independently of the mean() routine."""
    dist = Binomial(n, p)
    expectation = sum(k * dist.pmf_scalar(k) for k in range(n + 1))
    assert dist.mean() == pytest.approx(expectation, **ITERATIVE)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", PARAMS)
@pytest.mark.parametrize("t", [-1.0, -0.1, 0.0, 0.1, 0.5])
def test_mgf_matches_closed_form(t, n, p):
    assert Binomial(n, p).mgf_scalar(t) == pytest.approx(binom_mgf(t, n, p), **EXACT)


@pytest.mark.parametrize("n, p", [(10, 0.3), (5, 0.5), (20, 0.1)])
@pytest.mark.parametrize("t", [-1.0, -0.1, 0.0, 0.1, 0.5])
def test_cgf_matches_closed_form(t, n, p):
    assert Binomial(n, p).cgf_scalar(t) == pytest.approx(
        n * math.log(1 - p + p * math.exp(t)), **EXACT
    )


@pytest.mark.parametrize("n, p", PARAMS)
def test_mgf_at_zero_is_one(n, p):
    dist = Binomial(n, p)
    assert dist.mgf_scalar(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf_scalar(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n, p", [(10, 0.3), (5, 0.5)])
def test_cgf_is_log_of_mgf(n, p):
    dist = Binomial(n, p)
    for t in (-0.4, 0.2, 0.7):
        assert dist.cgf_scalar(t) == pytest.approx(
            math.log(dist.mgf_scalar(t)), **EXACT
        )


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", PARAMS)
def test_classmethods_agree_with_instances(n, p):
    dist = Binomial(n, p)
    assert Binomial._pmf_scalar(2, n, p) == pytest.approx(dist.pmf_scalar(2), **EXACT)
    assert Binomial._cdf_scalar(2, n, p) == pytest.approx(
        dist.cdf_scalar(2), **ITERATIVE
    )
    assert Binomial._mean(n, p) == pytest.approx(dist.mean(), **EXACT)
    assert Binomial._variance(n, p) == pytest.approx(dist.variance(), **EXACT)
    assert Binomial._stddev(n, p) == pytest.approx(dist.stddev(), **EXACT)
    assert Binomial._mgf_scalar(0.1, n, p) == pytest.approx(
        dist.mgf_scalar(0.1), **EXACT
    )
    assert Binomial._cgf_scalar(0.1, n, p) == pytest.approx(
        dist.cgf_scalar(0.1), **EXACT
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, 10, -0.1)),
    ("_pmf_scalar", (1, 10, 1.5)),
    ("_cdf_scalar", (1, 10, -0.5)),
    ("_mean", (10, 2.0)),
    ("_variance", (10, -1.0)),
    ("_stddev", (10, 1.5)),
    ("_mgf_scalar", (0.1, 10, -0.2)),
    ("_cgf_scalar", (0.1, 10, 1.2)),
    ("_sample", (10, -0.3)),
])
def test_classmethods_reject_out_of_range_p(method_name, args):
    with pytest.raises(ValueError, match=r"p must be in the interval \[0, 1\]"):
        getattr(Binomial, method_name)(*args)


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, -10, 0.5)),
    ("_cdf_scalar", (1, -1, 0.5)),
    ("_mean", (-5, 0.5)),
    ("_sample", (-2, 0.5)),
])
def test_classmethods_reject_negative_n(method_name, args):
    with pytest.raises(ValueError, match="n must be a non-negative integer"):
        getattr(Binomial, method_name)(*args)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, p", [(10, 0.3), (5, 0.5), (20, 0.9)])
def test_sample_lies_within_the_support(n, p):
    dist = Binomial(n, p)
    for _ in range(300):
        value = dist.sample()
        assert isinstance(value, int)
        assert 0 <= value <= n


def test_sample_is_deterministic_at_the_boundaries():
    assert all(Binomial(7, 0.0).sample() == 0 for _ in range(50))
    assert all(Binomial(7, 1.0).sample() == 7 for _ in range(50))


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Binomial(10, 0.3).extra = 123
