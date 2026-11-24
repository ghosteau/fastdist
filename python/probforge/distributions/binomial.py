# python/probforge/distributions/binomial.py
try:
    from probforge import _probforge as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "probforge's compiled extension (_probforge) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

import math
import numpy as np
from typing import Sequence, SupportsFloat, Union
from numpy.typing import NDArray

class Binomial:
    """
    Binomial distribution.

    Represents the number of successes in `n` independent trials, each
    succeeding with probability `p`. Supports scalar evaluation of PMF, log
    PMF, CDF, MGF, CGF, and sampling.

        P(X = k) = C(n, k) * p^k * (1-p)^(n-k),  k = 0, 1, ..., n

    Parameters
    ----------
    n : int
        Number of trials (must be a non-negative integer).
    p : float
        Probability of success on each trial (must be in [0, 1]).

    Attributes
    ----------
    n : int
        Number of trials.
    p : float
        Probability of success.

    Notes
    -----
    For sampling *without* replacement, where the trials are not independent,
    use `Hypergeometric` instead.

    The PMF is evaluated through `lgamma`, so a large `n` does not overflow the
    binomial coefficient even though the coefficient itself would.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_n", "_p")

    def __init__(self, n: int, p: SupportsFloat):
        Binomial._validate_params(n=n, p=p)
        self._n = int(n)
        self._p = float(p)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        self._validate_params(n=value)
        self._n = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._validate_params(p=value)
        self._p = float(value)

    def __repr__(self):
        return f"Binomial(n={self.n}, p={self.p})"

    @staticmethod
    def _validate_params(n: Union[int, None] = None, p: Union[SupportsFloat, None] = None) -> None:
        """Internal validation shared by all methods."""
        if n is not None:
            if not isinstance(n, int):
                raise TypeError("n must be an integer")
            if n < 0:
                raise ValueError("n must be a non-negative integer")

        if p is not None:
            if not isinstance(p, (int, float)):
                raise TypeError("p must be a real number")
            if not math.isfinite(p):
                raise ValueError("p must be finite")
            if not 0 <= p <= 1:
                raise ValueError("p must be in the interval [0, 1]")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        """
        Validate the evaluation points passed to this distribution's methods.

        Parameters
        ----------
        x : int or None, optional
            Point at which to evaluate. Skipped when None.
        t : float or None, optional
            Point at which to evaluate. Skipped when None.

        Raises
        ------
        TypeError
            If `x` is not an integer. If `t` is not a real number.

        Notes
        -----
        A value outside the support is not an error; it yields 0 or a
        boundary probability from the core, matching numpy's elementwise
        behavior.
        """
        if x is not None and not isinstance(x, int):
            raise TypeError("x must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def logpmf_scalar(self, x: int) -> float:
        """
        Compute the Binomial log probability mass function (log PMF).

        Evaluated through lgamma, so a large n cannot overflow the
        binomial coefficient the mass is built from.

        Parameters
        ----------
        x : int
            Success count at which to evaluate.

        Returns
        -------
        float
            Natural log of the mass at x; -inf outside the support.

        Raises
        ------
        TypeError
            If `x` is not an integer.
        """
        self._validate_inputs(x=x)
        return _core.binomial_logpmf_scalar(x, self.n, self.p)

    def pmf_scalar(self, x: int) -> float:
        """
        Compute the Binomial probability mass function (PMF).

        Parameters
        ----------
        x : int
            Success count at which to evaluate.

        Returns
        -------
        float
            C(n, k) * p^k * (1-p)^(n-k).

        Raises
        ------
        TypeError
            If `x` is not an integer.
        """
        self._validate_inputs(x=x)
        return _core.binomial_pmf_scalar(x, self.n, self.p)

    def cdf_scalar(self, x: int) -> float:
        """
        Compute the Binomial cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : int
            Success count at which to evaluate.

        Returns
        -------
        float
            the sum of the mass up to and including k.

        Raises
        ------
        TypeError
            If `x` is not an integer.
        """
        self._validate_inputs(x=x)
        return _core.binomial_cdf_scalar(x, self.n, self.p)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            n * p.
        """
        return _core.binomial_mean(self.n, self.p)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            n * p * (1 - p).
        """
        return _core.binomial_variance(self.n, self.p)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.binomial_stddev(self.n, self.p)

    def mgf_scalar(self, t: SupportsFloat) -> float:
        """
        Compute the Binomial moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            ((1 - p) + p*e^t)^n.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.binomial_mgf_scalar(float(t), self.n, self.p)

    def cgf_scalar(self, t: SupportsFloat) -> float:
        """
        Compute the Binomial cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            n * log((1 - p) + p*e^t).

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.binomial_cgf_scalar(float(t), self.n, self.p)

    def sample(self) -> int:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            One draw from Binomial(n, p).

        Notes
        -----
        Seed the shared engine with `probforge.seed` for reproducible draws.
        """
        return _core.binomial_sample(self.n, self.p)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _logpmf_scalar(cls, x: int, n: int, p: SupportsFloat) -> float:
        """
        Compute the Binomial log probability mass function (log PMF).

        Evaluated through lgamma, so a large n cannot overflow the
        binomial coefficient the mass is built from.

        Parameters
        ----------
        x : int
            Success count at which to evaluate.
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            Natural log of the mass at x; -inf outside the support.

        Raises
        ------
        TypeError
            If `x` is not an integer. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(x=x)
        return _core.binomial_logpmf_scalar(x, n, float(p))

    @classmethod
    def _pmf_scalar(cls, x: int, n: int, p: SupportsFloat) -> float:
        """
        Compute the Binomial probability mass function (PMF).

        Parameters
        ----------
        x : int
            Success count at which to evaluate.
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            C(n, k) * p^k * (1-p)^(n-k).

        Raises
        ------
        TypeError
            If `x` is not an integer. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(x=x)
        return _core.binomial_pmf_scalar(x, n, float(p))

    @classmethod
    def _cdf_scalar(cls, x: int, n: int, p: SupportsFloat) -> float:
        """
        Compute the Binomial cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : int
            Success count at which to evaluate.
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            the sum of the mass up to and including k.

        Raises
        ------
        TypeError
            If `x` is not an integer. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(x=x)
        return _core.binomial_cdf_scalar(x, n, float(p))

    @classmethod
    def _mean(cls, n: int, p: SupportsFloat) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            n * p.

        Raises
        ------
        TypeError
            If a parameter is of the wrong type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        return _core.binomial_mean(n, float(p))

    @classmethod
    def _variance(cls, n: int, p: SupportsFloat) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            n * p * (1 - p).

        Raises
        ------
        TypeError
            If a parameter is of the wrong type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        return _core.binomial_variance(n, float(p))

    @classmethod
    def _stddev(cls, n: int, p: SupportsFloat) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            Square root of the variance.

        Raises
        ------
        TypeError
            If a parameter is of the wrong type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        return _core.binomial_stddev(n, float(p))

    @classmethod
    def _mgf_scalar(cls, t: SupportsFloat, n: int, p: SupportsFloat) -> float:
        """
        Compute the Binomial moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            ((1 - p) + p*e^t)^n.

        Raises
        ------
        TypeError
            If `t` is not a real number. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(t=t)
        return _core.binomial_mgf_scalar(float(t), n, float(p))

    @classmethod
    def _cgf_scalar(cls, t: SupportsFloat, n: int, p: SupportsFloat) -> float:
        """
        Compute the Binomial cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        float
            n * log((1 - p) + p*e^t).

        Raises
        ------
        TypeError
            If `t` is not a real number. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(t=t)
        return _core.binomial_cgf_scalar(float(t), n, float(p))

    @classmethod
    def _sample(cls, n: int, p: SupportsFloat) -> int:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        n : int
            Number of trials (must be non-negative).
        p : float
            Probability of success on each trial, in [0, 1].

        Returns
        -------
        int
            One draw from Binomial(n, p).

        Raises
        ------
        TypeError
            If a parameter is of the wrong type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Seed the shared engine with `probforge.seed` for reproducible draws.
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(n=n, p=p)
        return _core.binomial_sample(n, float(p))
