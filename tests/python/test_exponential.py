"""
Numeric tests for the Exponential distribution against closed-form references.

`lambda_` is the rate parameter: mean = 1/lambda, f(x) = lambda e^(-lambda x).
"""

import math

import numpy as np
import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.exponential import Exponential


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def exp_pdf(x: float, lam: float) -> float:
    return lam * math.exp(-lam * x)


def exp_cdf(x: float, lam: float) -> float:
    return 1.0 - math.exp(-lam * x)


def exp_mgf(t: float, lam: float) -> float:
    """M(t) = lambda / (lambda - t), valid for t < lambda"""
    return lam / (lam - t)


RATES = [0.5, 1.0, 2.0, 5.0]
XS = [0.0, 0.25, 1.0, 3.0, 7.5]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameter():
    assert Exponential(lambda_=2.0).lambda_ == 2.0


@pytest.mark.parametrize("lam", [0.0, -0.1, -5.0])
def test_init_rejects_non_positive_rate(lam):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        Exponential(lambda_=lam)


@pytest.mark.parametrize("bad", ["2.0", None, object()])
def test_init_rejects_non_real_rate(bad):
    with pytest.raises(TypeError, match="lambda_ must be a real number"):
        Exponential(lambda_=bad)


def test_repr():
    assert repr(Exponential(lambda_=2.0)) == "Exponential(lambda_=2.0)"


def test_rate_setter_updates_and_validates():
    dist = Exponential(lambda_=2.0)
    dist.lambda_ = 4.0
    assert dist.lambda_ == 4.0
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        dist.lambda_ = 0.0


# ---------------------------------------------------------------------------
# Scalar PDF / CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("x", XS)
def test_pdf_matches_closed_form(x, lam):
    assert Exponential(lam).pdf(x) == pytest.approx(exp_pdf(x, lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("x", XS)
def test_cdf_matches_closed_form(x, lam):
    assert Exponential(lam).cdf(x) == pytest.approx(exp_cdf(x, lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
def test_pdf_at_zero_equals_the_rate(lam):
    assert Exponential(lam).pdf(0.0) == pytest.approx(lam, **EXACT)


@pytest.mark.parametrize("lam", RATES)
def test_cdf_is_monotonic_and_bounded(lam):
    dist = Exponential(lam)
    values = [dist.cdf(x) for x in (0.0, 0.1, 0.5, 1.0, 4.0, 20.0)]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("lam", RATES)
def test_cdf_starts_at_zero_and_saturates(lam):
    dist = Exponential(lam)
    assert dist.cdf(0.0) == pytest.approx(0.0, abs=1e-15)
    assert dist.cdf(500.0 / lam) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("lam", RATES)
def test_memorylessness(lam):
    """P(X > s + t | X > s) == P(X > t), the defining property."""
    dist = Exponential(lam)
    s, t = 1.5, 2.5
    survival = lambda z: 1.0 - dist.cdf(z)
    assert survival(s + t) / survival(s) == pytest.approx(survival(t), **ITERATIVE)


# ---------------------------------------------------------------------------
# Array API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_pdf_array_matches_scalar_evaluation(lam):
    dist = Exponential(lam)
    xs = [0.0, 0.25, 1.0, 3.0, 7.5]
    result = dist.pdf(xs)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (len(xs),)
    np.testing.assert_allclose(result, [exp_pdf(x, lam) for x in xs], rtol=1e-12)


@pytest.mark.parametrize("lam", RATES)
def test_cdf_array_matches_scalar_evaluation(lam):
    dist = Exponential(lam)
    xs = [0.0, 0.25, 1.0, 3.0, 7.5]
    np.testing.assert_allclose(
        dist.cdf(xs), [exp_cdf(x, lam) for x in xs], rtol=1e-12
    )


def test_array_accepts_numpy_input():
    dist = Exponential(2.0)
    xs = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(dist.pdf(xs), [exp_pdf(x, 2.0) for x in xs], rtol=1e-12)


def test_empty_array_returns_empty_array():
    result = Exponential(2.0).pdf([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_step_size_offsets_each_element_by_its_index():
    """With step_size s, element i is evaluated at x[i] + s*i."""
    dist = Exponential(2.0)
    result = dist.pdf([0.0, 0.0, 0.0], 0.5)
    np.testing.assert_allclose(
        result, [exp_pdf(0.0, 2.0), exp_pdf(0.5, 2.0), exp_pdf(1.0, 2.0)], rtol=1e-12
    )


def test_step_size_zero_is_a_plain_evaluation():
    dist = Exponential(2.0)
    np.testing.assert_allclose(dist.pdf([0.5, 1.0], 0), dist.pdf([0.5, 1.0]), rtol=1e-15)


def test_array_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        Exponential(2.0).pdf([[1.0, 2.0], [3.0, 4.0]])


def test_array_rejects_non_numeric_input():
    with pytest.raises(TypeError, match="must be numeric"):
        Exponential(2.0).pdf(["a", "b"])


def test_rejects_none_input():
    with pytest.raises(TypeError, match="must not be None"):
        Exponential(2.0).pdf(None)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_moments_match_closed_forms(lam):
    dist = Exponential(lam)
    assert dist.mean() == pytest.approx(1.0 / lam, **EXACT)
    assert dist.variance() == pytest.approx(1.0 / lam ** 2, **EXACT)
    assert dist.stddev() == pytest.approx(1.0 / lam, **EXACT)


@pytest.mark.parametrize("lam, override", [(2.0, 4.0), (1.0, 0.5)])
def test_moment_parameter_override(lam, override):
    """mean/variance/stddev accept an explicit rate that overrides the instance."""
    dist = Exponential(lam)
    assert dist.mean(override) == pytest.approx(1.0 / override, **EXACT)
    assert dist.variance(override) == pytest.approx(1.0 / override ** 2, **EXACT)
    assert dist.stddev(override) == pytest.approx(1.0 / override, **EXACT)
    assert dist.lambda_ == lam  # the override must not mutate the instance


@pytest.mark.parametrize("lam", RATES)
def test_moment_override_validates(lam):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        Exponential(lam).mean(-1.0)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("t_frac", [-2.0, -0.5, 0.0, 0.25, 0.5])
def test_mgf_matches_closed_form(t_frac, lam):
    t = t_frac * lam  # keep t strictly below lambda
    assert Exponential(lam).mgf(t) == pytest.approx(exp_mgf(t, lam), **EXACT)


@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("t_frac", [-2.0, -0.5, 0.0, 0.25, 0.5])
def test_cgf_matches_closed_form(t_frac, lam):
    t = t_frac * lam
    assert Exponential(lam).cgf(t) == pytest.approx(
        math.log(exp_mgf(t, lam)), **EXACT
    )


@pytest.mark.parametrize("lam", RATES)
def test_mgf_at_zero_is_one(lam):
    dist = Exponential(lam)
    assert dist.mgf(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("lam", [1.0, 2.0])
def test_mgf_array_matches_scalar_evaluation(lam):
    dist = Exponential(lam)
    ts = [-1.0, -0.25, 0.0, 0.25 * lam]
    np.testing.assert_allclose(
        dist.mgf(ts), [exp_mgf(t, lam) for t in ts], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
@pytest.mark.parametrize("x", [0.25, 1.0, 3.0])
def test_classmethods_agree_with_instances(x, lam):
    dist = Exponential(lam)
    assert Exponential._pdf_scalar(x, lam) == pytest.approx(dist.pdf(x), **EXACT)
    assert Exponential._cdf_scalar(x, lam) == pytest.approx(dist.cdf(x), **EXACT)
    assert Exponential._mgf_scalar(0.1, lam) == pytest.approx(dist.mgf(0.1), **EXACT)
    assert Exponential._cgf_scalar(0.1, lam) == pytest.approx(dist.cgf(0.1), **EXACT)


@pytest.mark.parametrize("lam", [1.0, 2.0])
def test_cpu_batch_classmethods_match_scalars(lam):
    xs = [0.0, 0.5, 1.0, 4.0]
    np.testing.assert_allclose(
        Exponential._pdf_cpu(xs, lam), [exp_pdf(x, lam) for x in xs], rtol=1e-12
    )
    np.testing.assert_allclose(
        Exponential._cdf_cpu(xs, lam), [exp_cdf(x, lam) for x in xs], rtol=1e-12
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pdf_scalar", (1.0, 0.0)),
    ("_pdf_scalar", (1.0, -2.0)),
    ("_cdf_scalar", (1.0, 0.0)),
    ("_mgf_scalar", (0.1, -1.0)),
    ("_cgf_scalar", (0.1, 0.0)),
])
def test_classmethods_reject_non_positive_rate(method_name, args):
    with pytest.raises(ValueError, match="lambda_ must be positive"):
        getattr(Exponential, method_name)(*args)


def test_is_cuda_available_returns_bool():
    assert isinstance(Exponential.is_cuda_available(), bool)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", RATES)
def test_sample_is_positive_and_finite(lam):
    dist = Exponential(lam)
    for _ in range(200):
        value = dist.sample()
        assert math.isfinite(value)
        assert value >= 0.0


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Exponential(2.0).extra = 123
