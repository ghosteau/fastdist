"""
Numeric tests for the continuous Uniform distribution against closed-form
references.

The support is the interval [a, b], so f(x) = 1/(b-a) inside it and 0 outside.
"""

import math

import numpy as np
import pytest

from conftest import EXACT, ITERATIVE
from fastdist.distributions.uniform import Uniform


# ---------------------------------------------------------------------------
# Closed-form references
# ---------------------------------------------------------------------------

def uniform_pdf(x: float, a: float, b: float) -> float:
    return 1.0 / (b - a) if a <= x <= b else 0.0


def uniform_cdf(x: float, a: float, b: float) -> float:
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (x - a) / (b - a)


def uniform_mgf(t: float, a: float, b: float) -> float:
    """M(t) = (e^(tb) - e^(ta)) / (t(b-a)); the t=0 singularity is removable."""
    if t == 0.0:
        return 1.0
    return (math.exp(t * b) - math.exp(t * a)) / (t * (b - a))


PARAMS = [(0.0, 1.0), (1.0, 3.0), (-2.0, 2.0), (-5.0, -1.0), (0.5, 10.0)]


# ---------------------------------------------------------------------------
# Constructor, properties, representation
# ---------------------------------------------------------------------------

def test_init_valid_parameters():
    dist = Uniform(a=1.0, b=3.0)
    assert dist.a == 1.0
    assert dist.b == 3.0


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (3.0, 1.0), (0.0, -1.0)])
def test_init_rejects_non_increasing_bounds(a, b):
    with pytest.raises(ValueError, match="a must be less than b"):
        Uniform(a=a, b=b)


@pytest.mark.parametrize("bad", ["1.0", object()])
def test_init_rejects_non_real_bounds(bad):
    with pytest.raises(TypeError, match="must be a real number"):
        Uniform(a=bad, b=3.0)


def test_init_rejects_none_bound():
    """
    None is rejected, but only incidentally: _validate_params skips its checks
    when a bound is None, so the failure surfaces from float() rather than as
    the intended "a must be a real number".
    """
    with pytest.raises(TypeError):
        Uniform(a=None, b=3.0)


def test_repr():
    assert repr(Uniform(a=1.0, b=3.0)) == "Uniform(a=1.0, b=3.0)"


def test_property_setters_update_values():
    dist = Uniform(a=1.0, b=3.0)
    dist.a = 0.0
    dist.b = 5.0
    assert dist.a == 0.0
    assert dist.b == 5.0


# KNOWN BUG: each setter validates only the bound being assigned, never the
# a < b relationship against the other one. Uniform(1.0, 3.0) can be driven to
# a = 10.0, b = -10.0 -- a state the constructor rejects outright. Once there,
# pdf, cdf, mean, variance and sample all return nan rather than raising, so the
# corruption propagates silently.
CROSS_BOUND_BUG = "setters do not re-validate a < b against the other bound"


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason=CROSS_BOUND_BUG)
def test_a_setter_rejects_value_above_b():
    dist = Uniform(a=1.0, b=3.0)
    with pytest.raises(ValueError, match="a must be less than b"):
        dist.a = 10.0


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason=CROSS_BOUND_BUG)
def test_b_setter_rejects_value_below_a():
    dist = Uniform(a=1.0, b=3.0)
    with pytest.raises(ValueError, match="a must be less than b"):
        dist.b = -10.0


@pytest.mark.known_bug
@pytest.mark.xfail(strict=True, reason=CROSS_BOUND_BUG + "; results become nan")
def test_instance_stays_usable_after_setter_assignments():
    dist = Uniform(a=1.0, b=3.0)
    try:
        dist.a = 10.0
    except ValueError:
        return  # rejecting the assignment is the correct behaviour
    assert math.isfinite(dist.mean())


# ---------------------------------------------------------------------------
# Scalar PDF / CDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_pdf_is_constant_inside_the_support(a, b):
    dist = Uniform(a, b)
    height = 1.0 / (b - a)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = a + frac * (b - a)
        assert dist.pdf(x) == pytest.approx(height, **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_pdf_is_zero_outside_the_support(a, b):
    dist = Uniform(a, b)
    assert dist.pdf(a - 1.0) == pytest.approx(0.0, abs=1e-15)
    assert dist.pdf(b + 1.0) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("a, b", PARAMS)
def test_pdf_integrates_to_one(a, b):
    """The density is constant, so the integral is just height * width."""
    dist = Uniform(a, b)
    assert dist.pdf((a + b) / 2) * (b - a) == pytest.approx(1.0, **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_matches_closed_form(a, b):
    dist = Uniform(a, b)
    for frac in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        x = a + frac * (b - a)
        assert dist.cdf(x) == pytest.approx(uniform_cdf(x, a, b), **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_saturates_at_the_bounds(a, b):
    dist = Uniform(a, b)
    assert dist.cdf(a) == pytest.approx(0.0, abs=1e-15)
    assert dist.cdf(b) == pytest.approx(1.0, **EXACT)
    assert dist.cdf(a - 10.0) == pytest.approx(0.0, abs=1e-15)
    assert dist.cdf(b + 10.0) == pytest.approx(1.0, **EXACT)


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_is_monotonic_and_bounded(a, b):
    dist = Uniform(a, b)
    xs = [a + frac * (b - a) for frac in (-0.5, 0.0, 0.3, 0.6, 1.0, 1.5)]
    values = [dist.cdf(x) for x in xs]
    assert values == sorted(values)
    assert all(0.0 <= v <= 1.0 for v in values)


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_at_the_midpoint_is_one_half(a, b):
    assert Uniform(a, b).cdf((a + b) / 2) == pytest.approx(0.5, **EXACT)


# ---------------------------------------------------------------------------
# Array API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_pdf_array_matches_scalar_evaluation(a, b):
    dist = Uniform(a, b)
    xs = [a - 1.0, a, (a + b) / 2, b, b + 1.0]
    result = dist.pdf(xs)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result, [uniform_pdf(x, a, b) for x in xs], rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("a, b", PARAMS)
def test_cdf_array_matches_scalar_evaluation(a, b):
    dist = Uniform(a, b)
    xs = [a - 1.0, a, (a + b) / 2, b, b + 1.0]
    np.testing.assert_allclose(
        dist.cdf(xs), [uniform_cdf(x, a, b) for x in xs], rtol=1e-12, atol=1e-15
    )


def test_array_accepts_numpy_input():
    dist = Uniform(1.0, 3.0)
    xs = np.array([0.0, 2.0, 5.0])
    np.testing.assert_allclose(dist.cdf(xs), [0.0, 0.5, 1.0], rtol=1e-12, atol=1e-15)


def test_empty_array_returns_empty_array():
    result = Uniform(1.0, 3.0).pdf([])
    assert isinstance(result, np.ndarray)
    assert result.size == 0


def test_step_size_offsets_each_element_by_its_index():
    """With step_size s, element i is evaluated at x[i] + s*i."""
    dist = Uniform(0.0, 4.0)
    result = dist.cdf([0.0, 0.0, 0.0], 1.0)
    np.testing.assert_allclose(result, [0.0, 0.25, 0.5], rtol=1e-12, atol=1e-15)


def test_array_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        Uniform(1.0, 3.0).pdf([[1.0, 2.0], [3.0, 4.0]])


def test_array_rejects_non_numeric_input():
    with pytest.raises(TypeError, match="must be numeric"):
        Uniform(1.0, 3.0).pdf(["a", "b"])


def test_rejects_none_input():
    with pytest.raises(TypeError, match="must not be None"):
        Uniform(1.0, 3.0).pdf(None)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_moments_match_closed_forms(a, b):
    dist = Uniform(a, b)
    assert dist.mean() == pytest.approx((a + b) / 2.0, **EXACT)
    assert dist.variance() == pytest.approx((b - a) ** 2 / 12.0, **EXACT)
    assert dist.stddev() == pytest.approx((b - a) / math.sqrt(12.0), **EXACT)


@pytest.mark.parametrize("a, b", [(0.0, 10.0), (-1.0, 1.0)])
def test_moment_parameter_override(a, b):
    dist = Uniform(1.0, 3.0)
    assert dist.mean(a, b) == pytest.approx((a + b) / 2.0, **EXACT)
    assert dist.variance(a, b) == pytest.approx((b - a) ** 2 / 12.0, **EXACT)
    assert dist.stddev(a, b) == pytest.approx((b - a) / math.sqrt(12.0), **EXACT)
    assert (dist.a, dist.b) == (1.0, 3.0)  # override must not mutate the instance


def test_moment_override_validates():
    with pytest.raises(ValueError, match="a must be less than b"):
        Uniform(1.0, 3.0).mean(5.0, 2.0)


# ---------------------------------------------------------------------------
# MGF / CGF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
@pytest.mark.parametrize("t", [-1.0, -0.25, 0.25, 1.0])
def test_mgf_matches_closed_form(t, a, b):
    assert Uniform(a, b).mgf(t) == pytest.approx(uniform_mgf(t, a, b), **ITERATIVE)


@pytest.mark.parametrize("a, b", PARAMS)
def test_mgf_at_zero_is_one(a, b):
    """t = 0 is a removable 0/0 singularity in the closed form."""
    dist = Uniform(a, b)
    assert dist.mgf(0.0) == pytest.approx(1.0, **EXACT)
    assert dist.cgf(0.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a, b", PARAMS)
@pytest.mark.parametrize("t", [-1.0, -0.25, 0.25, 1.0])
def test_cgf_is_log_of_mgf(t, a, b):
    dist = Uniform(a, b)
    assert dist.cgf(t) == pytest.approx(math.log(dist.mgf(t)), **ITERATIVE)


def test_mgf_array_matches_scalar_evaluation():
    dist = Uniform(1.0, 3.0)
    ts = [-1.0, -0.25, 0.0, 0.25, 1.0]
    np.testing.assert_allclose(
        dist.mgf(ts), [uniform_mgf(t, 1.0, 3.0) for t in ts], rtol=1e-10
    )


# ---------------------------------------------------------------------------
# Classmethods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_classmethods_agree_with_instances(a, b):
    dist = Uniform(a, b)
    mid = (a + b) / 2
    assert Uniform._pdf_scalar(mid, a, b) == pytest.approx(dist.pdf(mid), **EXACT)
    assert Uniform._cdf_scalar(mid, a, b) == pytest.approx(dist.cdf(mid), **EXACT)
    assert Uniform._mgf_scalar(0.25, a, b) == pytest.approx(
        dist.mgf(0.25), **ITERATIVE
    )
    assert Uniform._cgf_scalar(0.25, a, b) == pytest.approx(
        dist.cgf(0.25), **ITERATIVE
    )


@pytest.mark.parametrize("a, b", [(0.0, 1.0), (1.0, 3.0)])
def test_cpu_batch_classmethods_match_scalars(a, b):
    xs = [a - 0.5, a, (a + b) / 2, b, b + 0.5]
    np.testing.assert_allclose(
        Uniform._pdf_cpu(xs, a, b),
        [uniform_pdf(x, a, b) for x in xs],
        rtol=1e-12, atol=1e-15,
    )
    np.testing.assert_allclose(
        Uniform._cdf_cpu(xs, a, b),
        [uniform_cdf(x, a, b) for x in xs],
        rtol=1e-12, atol=1e-15,
    )


@pytest.mark.parametrize("method_name, args", [
    ("_pdf_scalar", (1.0, 3.0, 1.0)),
    ("_cdf_scalar", (1.0, 5.0, 2.0)),
    ("_mgf_scalar", (0.25, 3.0, 1.0)),
    ("_cgf_scalar", (0.25, 3.0, 1.0)),
])
def test_classmethods_reject_non_increasing_bounds(method_name, args):
    with pytest.raises(ValueError, match="a must be less than b"):
        getattr(Uniform, method_name)(*args)


def test_is_cuda_available_returns_bool():
    assert isinstance(Uniform.is_cuda_available(), bool)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", PARAMS)
def test_sample_lies_within_the_support(a, b):
    dist = Uniform(a, b)
    for _ in range(300):
        value = dist.sample()
        assert math.isfinite(value)
        assert a <= value <= b


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

def test_slots_prevent_dynamic_attributes():
    with pytest.raises(AttributeError):
        Uniform(1.0, 3.0).extra = 123
