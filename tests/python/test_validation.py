"""The validation contract, enforced uniformly across every distribution class.

The library has one rule at each layer, and these tests hold every class to it:

  * A *parameter* that cannot describe a distribution is refused at
    construction, with ValueError for a bad value and TypeError for a bad type.
    Nothing is constructed, so no later call can return nonsense.
  * An *input* x that is not finite is not an error. It yields nan, matching
    what the C++ core returns and what numpy does elementwise, so a single bad
    value in an array does not abort the whole call.

NaN is the case worth testing explicitly: every comparison against it is False,
so a range check like `p < 0 or p > 1` passes it through. Each class needs an
explicit finiteness check, and these tests fail if one loses it.
"""

import math

import pytest

from fastdist import (Bernoulli, Beta, Binomial, ChiSquare, DiscreteUniform, Exponential,
                      Gamma, Geometric, NegativeBinomial, Normal, Poisson, Uniform)

NAN = float("nan")
INF = float("inf")

# (class, valid args, index of a real-valued parameter, out-of-range value)
REAL_PARAM_CASES = [
    (Bernoulli, (0.3,), 0, 1.5),
    (Beta, (2.0, 5.0), 0, 0.0),
    (Beta, (2.0, 5.0), 1, -1.0),
    (Binomial, (10, 0.3), 1, 1.5),
    (ChiSquare, (6.0,), 0, 0.0),
    (Exponential, (2.0,), 0, 0.0),
    (Gamma, (3.0, 2.0), 0, 0.0),
    (Gamma, (3.0, 2.0), 1, -1.0),
    (Geometric, (0.25,), 0, 0.0),
    (NegativeBinomial, (4, 0.3), 1, 1.5),
    (Normal, (0.0, 1.0), 1, 0.0),
    (Poisson, (4.0,), 0, 0.0),
    (Uniform, (0.0, 1.0), 0, 2.0),
]


def _with(args, index, value):
    out = list(args)
    out[index] = value
    return tuple(out)


@pytest.mark.parametrize("cls, args, index, bad", REAL_PARAM_CASES)
def test_non_finite_parameters_are_refused(cls, args, index, bad):
    """nan and inf must raise, not build a distribution that returns nan forever."""
    for value in (NAN, INF, -INF):
        with pytest.raises(ValueError):
            cls(*_with(args, index, value))


@pytest.mark.parametrize("cls, args, index, bad", REAL_PARAM_CASES)
def test_out_of_range_parameters_are_refused(cls, args, index, bad):
    with pytest.raises(ValueError):
        cls(*_with(args, index, bad))


@pytest.mark.parametrize("cls, args, index, bad", REAL_PARAM_CASES)
def test_non_numeric_parameters_are_refused(cls, args, index, bad):
    for value in ("0.5", None, [0.5]):
        with pytest.raises(TypeError):
            cls(*_with(args, index, value))


@pytest.mark.parametrize("cls, args", [
    (Binomial, (10, 0.3)),
    (NegativeBinomial, (4, 0.3)),
    (DiscreteUniform, (1, 6)),
])
def test_integer_parameters_reject_floats(cls, args):
    """These take counts; a float is a mistake, not something to round."""
    with pytest.raises(TypeError):
        cls(*(( 2.5,) + args[1:]))


def test_bounds_must_be_ordered():
    with pytest.raises(ValueError):
        Uniform(5.0, 1.0)
    with pytest.raises(ValueError):
        DiscreteUniform(6, 1)


# (a valid instance, a method taking x, a point inside the support)
INPUT_CASES = [
    (Normal(0.0, 1.0), "pdf", 0.5),
    (Normal(0.0, 1.0), "cdf", 0.5),
    (Exponential(2.0), "pdf", 0.5),
    (Exponential(2.0), "cdf", 0.5),
    (Poisson(4.0), "pmf", 3.0),
    (Beta(2.0, 5.0), "pdf_scalar", 0.5),
    (Gamma(3.0, 2.0), "pmf_scalar", 1.0),
    (ChiSquare(6.0), "pdf", 4.0),
]


@pytest.mark.parametrize("dist, method, good", INPUT_CASES)
def test_non_finite_input_yields_nan(dist, method, good):
    """A non-finite x is answered with nan rather than an exception."""
    fn = getattr(dist, method)
    assert math.isfinite(fn(good))
    for value in (NAN, INF, -INF):
        assert math.isnan(fn(value)), f"{type(dist).__name__}.{method}({value}) should be nan"


@pytest.mark.parametrize("dist, method, good", INPUT_CASES)
def test_non_numeric_input_raises(dist, method, good):
    fn = getattr(dist, method)
    for value in ("0.5", None):
        with pytest.raises(TypeError):
            fn(value)
