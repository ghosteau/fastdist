"""
Shared pytest configuration and reference implementations for the fastdist suite.

The suite exercises the *real* compiled extension. Every numeric assertion is
made against an independently derived closed form or a reference implementation
written here, rather than against a value captured from a previous run, so a
regression in the C++ or CUDA backend surfaces as a test failure instead of
passing silently.
"""

import math
import sys
from pathlib import Path

# Fall back to the source tree when the package has not been pip-installed.
try:
    import fastdist  # noqa: F401
except ImportError:  # pragma: no cover - exercised only in uninstalled checkouts
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------------
# Numeric tolerances
# ----------------------------------------------------------------------------
# Closed-form algebra (means, variances, MGFs) is accurate to machine epsilon.
EXACT = {"rel": 1e-12, "abs": 1e-15}

# Iterative routines (regularized incomplete gamma/beta) converge to EPS=1e-12
# as configured in include/fastdist/config.h; leave headroom above that.
ITERATIVE = {"rel": 1e-10, "abs": 1e-12}


# ----------------------------------------------------------------------------
# Reference implementations
# ----------------------------------------------------------------------------

def regularized_lower_gamma(a: float, x: float) -> float:
    """
    P(a, x), the regularized lower incomplete gamma function.

    Reference implementation using the standard series expansion for x < a+1
    and the Lentz continued fraction for x >= a+1 (Numerical Recipes 6.2).
    Used to validate the backend's Gamma and Chi-square CDFs.
    """
    if a <= 0.0 or x < 0.0:
        raise ValueError("a must be positive and x non-negative")
    if x == 0.0:
        return 0.0

    gln = math.lgamma(a)

    if x < a + 1.0:
        # Series representation
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-16:
                break
        return total * math.exp(-x + a * math.log(x) - gln)

    # Continued fraction representation (modified Lentz)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """
    I_x(a, b), the regularized incomplete beta function.

    Reference implementation using the continued fraction of Numerical Recipes
    6.4 with the standard symmetry transformation. Used to validate the
    backend's Beta CDF.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )

    # Strict comparison: at exactly the swap boundary with a == b, a non-strict
    # test would swap to the identical point and recurse forever.
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x)

    tiny = 1e-300
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, 1000):
        m2 = 2 * m

        numerator = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c

        numerator = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < 1e-15:
            break

    return front * h / a


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "known_bug: marks a test that documents a confirmed defect in the backend",
    )
