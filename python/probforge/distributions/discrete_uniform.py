# python/probforge/distributions/discrete_uniform.py
try:
    from probforge import _probforge as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "probforge's compiled extension (_probforge) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

import numpy as np
from typing import Sequence, Union
from numpy.typing import NDArray

class DiscreteUniform:
    """
    Discrete uniform distribution.

    Represents an integer drawn with equal probability from the inclusive
    range {a, ..., b} -- a fair die being the canonical case. Supports scalar
    evaluation of PMF, CDF, MGF, CGF, and sampling.

        P(X = x) = 1 / (b - a + 1),  x in {a, ..., b}

    Parameters
    ----------
    a : int
        Lower bound of the support, inclusive.
    b : int
        Upper bound of the support, inclusive (must be at least `a`).

    Attributes
    ----------
    a : int
        Lower bound of the support.
    b : int
        Upper bound of the support.

    Notes
    -----
    See `Uniform` for the continuous case on an interval.

    The support size is formed in floating point, so a range spanning the
    whole 32-bit integer range is handled rather than wrapping.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_a", "_b")

    def __init__(self, a: int, b: int):
        self._validate_params(a=a, b=b)
        self._a = int(a)
        self._b = int(b)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        # Validate against the current opposite bound so the a < b invariant
        # holds after every assignment; store as int, as __init__ does.
        self._validate_params(a=value, b=self._b)
        self._a = int(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._validate_params(a=self._a, b=value)
        self._b = int(value)

    def __repr__(self):
        return f"DiscreteUniform(a={self.a}, b={self.b})"

    @staticmethod
    def _validate_params(a: Union[int, None] = None, b: Union[int, None] = None) -> None:
        """Internal validation shared by all methods."""
        if a is not None:
            if not isinstance(a, int):
                raise TypeError("a must be an integer")
        if b is not None:
            if not isinstance(b, int):
                raise TypeError("b must be an integer")
        if a is not None and b is not None:
            if a >= b:
                raise ValueError("a must be less than b")

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
    def pmf(self, x: int) -> float:
        """
        Compute the DiscreteUniform probability mass function (PMF).

        Parameters
        ----------
        x : int
            Value at which to evaluate.

        Returns
        -------
        float
            1 / (b - a + 1).

        Raises
        ------
        TypeError
            If `x` is not an integer.
        """
        self._validate_inputs(x=x)
        return _core.discrete_uniform_pmf_scalar(x, self.a, self.b)

    def cdf(self, x: int) -> float:
        """
        Compute the DiscreteUniform cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : int
            Value at which to evaluate.

        Returns
        -------
        float
            (x - a + 1) / (b - a + 1).

        Raises
        ------
        TypeError
            If `x` is not an integer.
        """
        self._validate_inputs(x=x)
        return _core.discrete_uniform_cdf_scalar(x, self.a, self.b)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            (a + b) / 2.
        """
        return _core.discrete_uniform_mean(self.a, self.b)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            (n^2 - 1) / 12, where n = b - a + 1.
        """
        return _core.discrete_uniform_variance(self.a, self.b)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.discrete_uniform_stddev(self.a, self.b)

    def mgf(self, t: Union[int, float]) -> float:
        """
        Compute the DiscreteUniform moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            e^(ta) * (e^(tn) - 1) / (n * (e^t - 1)); 1 at t = 0.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.discrete_uniform_mgf_scalar(float(t), self.a, self.b)

    def cgf(self, t: Union[int, float]) -> float:
        """
        Compute the DiscreteUniform cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            log of the MGF.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.discrete_uniform_cgf_scalar(float(t), self.a, self.b)

    def sample(self) -> int:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            One draw, uniform over {a, ..., b}.

        Notes
        -----
        Seed the shared engine with `probforge.seed` for reproducible draws.
        """
        return _core.discrete_uniform_sample(self.a, self.b)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, x: int, a: int, b: int) -> float:
        """
        Compute the DiscreteUniform probability mass function (PMF).

        Parameters
        ----------
        x : int
            Value at which to evaluate.
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        float
            1 / (b - a + 1).

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
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(x=x)
        return _core.discrete_uniform_pmf_scalar(x, a, b)

    @classmethod
    def _cdf_scalar(cls, x: int, a: int, b: int) -> float:
        """
        Compute the DiscreteUniform cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : int
            Value at which to evaluate.
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        float
            (x - a + 1) / (b - a + 1).

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
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(x=x)
        return _core.discrete_uniform_cdf_scalar(x, a, b)

    @classmethod
    def _mean(cls, a: int, b: int) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        float
            (a + b) / 2.

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
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_mean(a, b)

    @classmethod
    def _variance(cls, a: int, b: int) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        float
            (n^2 - 1) / 12, where n = b - a + 1.

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
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_variance(a, b)

    @classmethod
    def _stddev(cls, a: int, b: int) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

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
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_stddev(a, b)

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], a: int, b: int) -> float:
        """
        Compute the DiscreteUniform moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        float
            e^(ta) * (e^(tn) - 1) / (n * (e^t - 1)); 1 at t = 0.

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
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(t=t)
        return _core.discrete_uniform_mgf_scalar(float(t), a, b)

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], a: int, b: int) -> float:
        """
        Compute the DiscreteUniform cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        float
            log of the MGF.

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
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(t=t)
        return _core.discrete_uniform_cgf_scalar(float(t), a, b)

    @classmethod
    def _sample(cls, a: int, b: int) -> int:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        a : int
            Lower bound of the support, inclusive.
        b : int
            Upper bound of the support, inclusive.

        Returns
        -------
        int
            One draw, uniform over {a, ..., b}.

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
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_sample(a, b)
