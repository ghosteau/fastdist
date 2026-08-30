"""Numeric tests for the Poisson distribution against closed-form references."""

import math

import numpy as np
import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.poisson import Poisson


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) = lambda^k e^(-lambda) / k!"""
    return lam ** k * math.exp(-lam) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def poisson_mgf(t: float, lam: float) -> float:
    """M(t) = exp(lambda (e^t - 1))"""
    return math.exp(lam * (math.exp(t) - 1))


RATES = [0.5, 1.0, 4.0, 10.0]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    assert Poisson(lambda_=4.0).lambda_ == 4.0


@pytest.mark.parametrize("lam", [0.0, -0.1, -5.0])
def test_init_rejects_non_positive_rate(lam):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        Poisson(lambda_=lam)


@pytest.mark.parametrize("bad", ["4.0", None, object()])
def test_init_rejects_non_real_rate(bad):
    with pytest.raises(TypeError, match="lambda_ must be a real number"):
        Poisson(lambda_=bad)


def test_repr():
    assert repr(Poisson(lambda_=4.0)) == "Poisson(lambda_=4.0)"


def test_rate_setter_updates_and_validates():
    dist = Poisson(lambda_=4.0)
    dist.lambda_ = 7.0
    assert dist.lambda_ == 7.0
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        dist.lambda_ = 0.0


# ---------------------------------------------------------------------------
# PMF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("k", [0, 1, 2, 5, 10, 20])
def test_pmf_matches_closed_form(k, lam):
    assert Poisson(lam).pmf(k) == pytest.approx(poisson_pmf(k, lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
def test_pmf_sums_to_one(lam):
    dist = Poisson(lam)
    total = sum(dist.pmf(k) for k in range(0, 200))
    assert total == pytest.approx(1.0, **ITERATIVE)


@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("k", [0, 3, 12])
def test_pmf_is_a_probability(k, lam):
    assert 0.0 <= Poisson(lam).pmf(k) <= 1.0


@pytest.mark.parametrize("lam", RATES)
def test_pmf_at_zero_is_exp_minus_lambda(lam):
    assert Poisson(lam).pmf(0) == pytest.approx(math.exp(-lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
def test_pmf_recurrence(lam):
    """P(k) = P(k-1) * lambda / k, independent of the factorial formulation."""
    dist = Poisson(lam)
    for k in range(1, 25):
        assert dist.pmf(k) == pytest.approx(dist.pmf(k - 1) * lam / k, **ITERATIVE)


# ---------------------------------------------------------------------------
# CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("k", [0, 1, 3, 8, 20])
def test_cdf_matches_summed_pmf(k, lam):
    assert Poisson(lam).cdf(k) == pytest.approx(poisson_cdf(k, lam), **ITERATIVE)


@pytest.mark.parametrize("lam", RATES)
def test_cdf_is_monotonic_and_bounded(lam):
    dist = Poisson(lam)
    values = [dist.cdf(k) for k in range(0, 40)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("lam", RATES)
def test_cdf_approaches_one_in_the_tail(lam):
    assert Poisson(lam).cdf(150) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Array API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_pmf_array_matches_scalar_evaluation(lam):
    dist = Poisson(lam)
    ks = [0, 1, 2, 5, 10]
    result = dist.pmf(ks)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [poisson_pmf(k, lam) for k in ks], rtol=1e-12)


@pytest.mark.parametrize("lam", RATES)
def test_cdf_array_matches_scalar_evaluation(lam):
    dist = Poisson(lam)
    ks = [0, 1, 3, 8]
    np.testing.assert_allclose(
        dist.cdf(ks), [poisson_cdf(k, lam) for k in ks], rtol=1e-10
    )


def test_array_accepts_numpy_input():
    dist = Poisson(4.0)
    ks = np.array([0, 2, 5])
    np.testing.assert_allclose(
        dist.pmf(ks), [poisson_pmf(int(k), 4.0) for k in ks], rtol=1e-12
    )


def test_empty_array_returns_empty_array():
    result = Poisson(4.0).pmf([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_step_size_offsets_each_element_by_its_index():
    """With step_size s, element i is evaluated at x[i] + s*i."""
    dist = Poisson(4.0)
    result = dist.pmf([0, 0, 0], 1)
    np.testing.assert_allclose(
        result,
        [poisson_pmf(0, 4.0), poisson_pmf(1, 4.0), poisson_pmf(2, 4.0)],
        rtol=1e-12,
    )


def test_step_size_must_be_an_integer():
    with pytest.raises(TypeError, match="step_size must be an integer"):
        Poisson(4.0).pmf([0, 1, 2], 0.5)


def test_array_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        Poisson(4.0).pmf([[1, 2], [3, 4]])


def test_array_rejects_non_numeric_input():
    with pytest.raises(TypeError, match="must be numeric"):
        Poisson(4.0).pmf(["a", "b"])


def test_rejects_none_input():
    with pytest.raises(TypeError, match="cannot be None"):
        Poisson(4.0).pmf(None)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_moments_match_closed_forms(lam):
    dist = Poisson(lam)
    assert dist.mean() == pytest.approx(lam, **EXACT)
    assert dist.variance() == pytest.approx(lam, **EXACT)
    assert dist.stddev() == pytest.approx(math.sqrt(lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
def test_mean_equals_pmf_weighted_sum(lam):
    dist = Poisson(lam)
    expectation = sum(k * dist.pmf(k) for k in range(0, 200))
    assert dist.mean() == pytest.approx(expectation, rel=1e-9)


@pytest.mark.parametrize("lam, override", [(4.0, 7.0), (1.0, 0.5)])
def test_moment_parameter_override(lam, override):
    dist = Poisson(lam)
    assert dist.mean(override) == pytest.approx(override, **EXACT)
    assert dist.variance(override) == pytest.approx(override, **EXACT)
    assert dist.stddev(override) == pytest.approx(math.sqrt(override), **EXACT)
    assert dist.lambda_ == lam  # the override must not mutate the instance


def test_moment_override_validates():
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        Poisson(4.0).mean(-1.0)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("t", [-1.0, -0.25, 0.0, 0.1, 0.5])
def test_mgf_matches_closed_form(t, lam):
    assert Poisson(lam).mgf(t) == pytest.approx(poisson_mgf(t, lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("t", [-1.0, -0.25, 0.0, 0.1, 0.5])
def test_cgf_matches_closed_form(t, lam):
    assert Poisson(lam).cgf(t) == pytest.approx(
        lam * (math.exp(t) - 1), **EXACT
    )


@pytest.mark.parametrize("lam", RATES)
def test_mgf_at_zero_is_one(lam):
    dist = Poisson(lam)
    assert dist.mgf(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("lam", [1.0, 4.0])
def test_cgf_is_log_of_mgf(lam):
    dist = Poisson(lam)
    for t in (-0.5, 0.1, 0.4):
        assert dist.cgf(t) == pytest.approx(math.log(dist.mgf(t)), **EXACT)


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_classmethods_agree_with_instances(lam):
    dist = Poisson(lam)
    assert Poisson._pmf_scalar(3, lam) == pytest.approx(dist.pmf(3), **EXACT)
    assert Poisson._cdf_scalar(3, lam) == pytest.approx(dist.cdf(3), **ITERATIVE)
    assert Poisson._mgf_scalar(0.1, lam) == pytest.approx(dist.mgf(0.1), **EXACT)
    assert Poisson._cgf_scalar(0.1, lam) == pytest.approx(dist.cgf(0.1), **EXACT)


@pytest.mark.parametrize("lam", [1.0, 4.0])
def test_cpu_batch_classmethods_match_scalars(lam):
    ks = [0, 1, 2, 5]
    np.testing.assert_allclose(
        Poisson._pmf_cpu(ks, lam), [poisson_pmf(k, lam) for k in ks], rtol=1e-12
    )
    np.testing.assert_allclose(
        Poisson._cdf_cpu(ks, lam), [poisson_cdf(k, lam) for k in ks], rtol=1e-10
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pmf_scalar", (1, 0.0)),
    ("_pmf_scalar", (1, -2.0)),
    ("_cdf_scalar", (1, 0.0)),
    ("_mgf_scalar", (0.1, -1.0)),
    ("_cgf_scalar", (0.1, 0.0)),
])
def test_classmethods_reject_non_positive_rate(method_name, args):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        getattr(Poisson, method_name)(*args)


def test_is_cuda_available_returns_bool():
    assert isinstance(Poisson.is_cuda_available(), bool)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_sample_is_a_non_negative_integer(lam):
    dist = Poisson(lam)
    for _ in range(300):
        value = dist.sample()
        assert isinstance(value, int)
        assert value >= 0


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Poisson(4.0).extra = 123
