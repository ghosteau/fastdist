"""
Numeric tests for the Geometric distribution against closed-form references.

This implementation uses the "number of trials until the first success"
convention: the support is k = 1, 2, 3, ... and P(X = k) = (1-p)^(k-1) p,
so the mean is 1/p.
"""

import math

import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.geometric import Geometric


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def geom_pmf(k: int, p: float) -> float:
    """P(X = k) = (1-p)^(k-1) p for k >= 1"""
    return (1 - p) ** (k - 1) * p


def geom_cdf(k: int, p: float) -> float:
    """F(k) = 1 - (1-p)^k for k >= 1"""
    return 1.0 - (1 - p) ** k


def geom_mgf(t: float, p: float) -> float:
    """M(t) = p e^t / (1 - (1-p) e^t), valid for t < -ln(1-p)"""
    return p * math.exp(t) / (1 - (1 - p) * math.exp(t))


PROBS = [0.1, 0.3, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    assert Geometric(p=0.3).p == 0.3


@pytest.mark.parametrize("p", [0.0, -0.1, 1.1, 2.0])
def test_init_rejects_out_of_range_p(p):
    with pytest.raises(ValueError, match=r"p must be in the interval \(0, 1\]"):
        Geometric(p=p)


@pytest.mark.parametrize("bad", ["0.5", None, object()])
def test_init_rejects_non_real_p(bad):
    with pytest.raises(TypeError, match="p must be a real number"):
        Geometric(p=bad)


def test_repr():
    assert repr(Geometric(p=0.3)) == "Geometric(p=0.3)"


def test_p_setter_updates_and_validates():
    dist = Geometric(p=0.3)
    dist.p = 0.8
    assert dist.p == 0.8
    with pytest.raises(ValueError, match=r"p must be in the interval \(0, 1\]"):
        dist.p = 0.0


# ---------------------------------------------------------------------------
# PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", PROBS)
@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_pmf_matches_closed_form(k, p):
    assert Geometric(p).pmf_scalar(k) == pytest.approx(geom_pmf(k, p), **EXACT)


@pytest.mark.parametrize("p", PROBS)
def test_pmf_is_zero_below_the_support(p):
    """The support starts at k = 1, so k = 0 has zero mass."""
    assert Geometric(p).pmf_scalar(0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.75])
def test_pmf_sums_to_one(p):
    dist = Geometric(p)
    total = sum(dist.pmf_scalar(k) for k in range(1, 2000))
    assert total == pytest.approx(1.0, **ITERATIVE)


@pytest.mark.parametrize("p", PROBS)
@pytest.mark.parametrize("k", [1, 4, 9])
def test_pmf_is_a_probability(k, p):
    assert 0.0 <= Geometric(p).pmf_scalar(k) <= 1.0


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
def test_pmf_is_decreasing(p):
    dist = Geometric(p)
    values = [dist.pmf_scalar(k) for k in range(1, 12)]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", PROBS)
@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_cdf_matches_closed_form(k, p):
    assert Geometric(p).cdf_scalar(k) == pytest.approx(geom_cdf(k, p), **ITERATIVE)


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.75])
@pytest.mark.parametrize("k", [1, 3, 7])
def test_cdf_matches_summed_pmf(k, p):
    dist = Geometric(p)
    assert dist.cdf_scalar(k) == pytest.approx(
        sum(dist.pmf_scalar(i) for i in range(1, k + 1)), **ITERATIVE
    )


@pytest.mark.parametrize("p", PROBS)
def test_cdf_is_monotonic_and_bounded(p):
    dist = Geometric(p)
    values = [dist.cdf_scalar(k) for k in range(1, 20)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", PROBS)
def test_moments_match_closed_forms(p):
    dist = Geometric(p)
    assert dist.mean() == pytest.approx(1.0 / p, **EXACT)
    assert dist.variance() == pytest.approx((1 - p) / p ** 2, **EXACT)
    assert dist.stddev() == pytest.approx(math.sqrt((1 - p) / p ** 2), **EXACT)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.75])
def test_mean_equals_pmf_weighted_sum(p):
    dist = Geometric(p)
    expectation = sum(k * dist.pmf_scalar(k) for k in range(1, 5000))
    assert dist.mean() == pytest.approx(expectation, rel=1e-8)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.3, 0.5, 0.75])
@pytest.mark.parametrize("t", [-1.0, -0.2, 0.0, 0.1])
def test_mgf_matches_closed_form(t, p):
    assert Geometric(p).mgf_scalar(t) == pytest.approx(geom_mgf(t, p), **EXACT)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.75])
@pytest.mark.parametrize("t", [-1.0, -0.2, 0.0, 0.1])
def test_cgf_matches_closed_form(t, p):
    assert Geometric(p).cgf_scalar(t) == pytest.approx(
        math.log(geom_mgf(t, p)), **EXACT
    )


@pytest.mark.parametrize("p", [0.3, 0.5, 0.75])
def test_mgf_at_zero_is_one(p):
    dist = Geometric(p)
    assert dist.mgf_scalar(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf_scalar(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("p", [0.3, 0.5])
def test_cgf_is_log_of_mgf(p):
    dist = Geometric(p)
    for t in (-0.5, -0.1, 0.05):
        assert dist.cgf_scalar(t) == pytest.approx(
            math.log(dist.mgf_scalar(t)), **EXACT
        )


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.1, 0.5, 0.75])
def test_classmethods_agree_with_instances(p):
    dist = Geometric(p)
    assert Geometric._pmf_scalar(3, p) == pytest.approx(dist.pmf_scalar(3), **EXACT)
    assert Geometric._cdf_scalar(3, p) == pytest.approx(
        dist.cdf_scalar(3), **ITERATIVE
    )
    assert Geometric._mean(p) == pytest.approx(dist.mean(), **EXACT)
    assert Geometric._variance(p) == pytest.approx(dist.variance(), **EXACT)
    assert Geometric._stddev(p) == pytest.approx(dist.stddev(), **EXACT)
    assert Geometric._mgf_scalar(0.1, p) == pytest.approx(
        dist.mgf_scalar(0.1), **EXACT
    )
    assert Geometric._cgf_scalar(0.1, p) == pytest.approx(
        dist.cgf_scalar(0.1), **EXACT
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, 0.0)),
    ("_pmf_scalar", (1, 1.5)),
    ("_cdf_scalar", (1, -0.2)),
    ("_mean", (0.0,)),
    ("_variance", (1.5,)),
    ("_stddev", (-1.0,)),
    ("_mgf_scalar", (0.1, 0.0)),
    ("_cgf_scalar", (0.1, 2.0)),
    ("_sample", (0.0,)),
])
def test_classmethods_reject_out_of_range_p(method_name, args):
    with pytest.raises(ValueError, match=r"p must be in the interval \(0, 1\]"):
        getattr(Geometric, method_name)(*args)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_sample_lies_within_the_support(p):
    dist = Geometric(p)
    for _ in range(300):
        value = dist.sample()
        assert isinstance(value, int)
        assert value >= 1


def test_sample_is_deterministic_for_certain_success():
    assert all(Geometric(1.0).sample() == 1 for _ in range(50))


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Geometric(0.3).extra = 123
