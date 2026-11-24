# python/fastdist/distributions/chi_square.py
try:
    from fastdist import _fastdist as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "fastdist's compiled extension (_fastdist) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

import math
import numpy as np
from typing import Sequence, Union
from numpy.typing import NDArray

class ChiSquare:
    """
    Chi-square distribution.

    Represents the distribution of a sum of `k` squared standard normal
    variables, which is what makes it the reference distribution for variance
    and goodness-of-fit tests. Supports scalar evaluation of PDF, CDF, MGF,
    CGF, and sampling.

        f(x) = x^(k/2-1) * exp(-x/2) / (2^(k/2) * Gamma(k/2)),  x >= 0

    Parameters
    ----------
    k : float
        Degrees of freedom (must be positive and finite).

    Attributes
    ----------
    k : float
        Degrees of freedom.

    Notes
    -----
    This is the Gamma distribution with shape ``k / 2`` and scale 2, and is
    implemented by delegating to it. `k` is a float rather than an int so
    non-integer degrees of freedom work.

    The CDF is the regularized lower incomplete gamma function, evaluated
    iteratively, so it is accurate to roughly 1e-12 rather than to machine
    epsilon.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_k",)

    def __init__(self, k: Union[int, float]):
        ChiSquare._validate_params(k=k)
        self._k = float(k)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._validate_params(k=value)
        self._k = float(value)

    def __repr__(self):
        return f"ChiSquare(k={self.k})"

    @staticmethod
    def _validate_params(k: Union[int, float]) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(k, (int, float)):
            raise TypeError("k must be a real number")
        if not math.isfinite(k):
            raise ValueError("k must be finite")
        if k <= 0:
            raise ValueError("k must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        """
        Validate the evaluation points passed to this distribution's methods.

        Parameters
        ----------
        x : float or None, optional
            Point at which to evaluate. Skipped when None.
        t : float or None, optional
            Point at which to evaluate. Skipped when None.

        Raises
        ------
        TypeError
            If `x` is not a real number. If `t` is not a real number.

        Notes
        -----
        A value outside the support is not an error; it yields 0 or a
        boundary probability from the core, matching numpy's elementwise
        behaviour.
        """
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pdf(self, x: Union[int, float]) -> float:
        """
        Compute the ChiSquare probability density function (PDF).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.

        Returns
        -------
        float
            x^(k/2-1) * exp(-x/2) / (2^(k/2) * Gamma(k/2)).

        Raises
        ------
        TypeError
            If `x` is not a real number.
        """
        self._validate_inputs(x=x)
        return _core.chi_square_pdf_scalar(float(x), self.k)

    def cdf(self, x: Union[int, float]) -> float:
        """
        Compute the ChiSquare cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.

        Returns
        -------
        float
            the regularized lower incomplete gamma P(k/2, x/2).

        Raises
        ------
        TypeError
            If `x` is not a real number.

        Notes
        -----
        Evaluated iteratively, so it is accurate to roughly 1e-12 rather than to
        machine epsilon.
        """
        self._validate_inputs(x=x)
        return _core.chi_square_cdf_scalar(float(x), self.k)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            k.
        """
        return _core.chi_square_mean(self.k)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            2 * k.
        """
        return _core.chi_square_variance(self.k)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.chi_square_stddev(self.k)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the ChiSquare moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            (1 - 2t)^(-k/2), defined for t < 1/2.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.chi_square_mgf_scalar(float(t), self.k)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the ChiSquare cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            -(k/2) * log(1 - 2t), defined for t < 1/2.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.chi_square_cgf_scalar(float(t), self.k)

    def sample(self) -> float:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            One draw from ChiSquare(k), via the equivalent Gamma(k/2, 2).

        Notes
        -----
        Seed the shared engine with `fastdist.seed` for reproducible draws.
        """
        return _core.chi_square_sample(self.k)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float], k: Union[int, float]) -> float:
        """
        Compute the ChiSquare probability density function (PDF).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        float
            x^(k/2-1) * exp(-x/2) / (2^(k/2) * Gamma(k/2)).

        Raises
        ------
        TypeError
            If `x` is not a real number. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(k=k)
        cls._validate_inputs(x=x)
        return _core.chi_square_pdf_scalar(float(x), float(k))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float], k: Union[int, float]) -> float:
        """
        Compute the ChiSquare cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        float
            the regularized lower incomplete gamma P(k/2, x/2).

        Raises
        ------
        TypeError
            If `x` is not a real number. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Evaluated iteratively, so it is accurate to roughly 1e-12 rather than to
        machine epsilon.
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(k=k)
        cls._validate_inputs(x=x)
        return _core.chi_square_cdf_scalar(float(x), float(k))

    @classmethod
    def _mean(cls, k: Union[int, float]) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        float
            k.

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
        cls._validate_params(k=k)
        return _core.chi_square_mean(float(k))

    @classmethod
    def _variance(cls, k: Union[int, float]) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        float
            2 * k.

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
        cls._validate_params(k=k)
        return _core.chi_square_variance(float(k))

    @classmethod
    def _stddev(cls, k: Union[int, float]) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        k : float
            Degrees of freedom (must be positive).

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
        cls._validate_params(k=k)
        return _core.chi_square_stddev(float(k))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], k: Union[int, float]) -> float:
        """
        Compute the ChiSquare moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        float
            (1 - 2t)^(-k/2), defined for t < 1/2.

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
        cls._validate_params(k=k)
        cls._validate_inputs(t=t)
        return _core.chi_square_mgf_scalar(float(t), float(k))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], k: Union[int, float]) -> float:
        """
        Compute the ChiSquare cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        float
            -(k/2) * log(1 - 2t), defined for t < 1/2.

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
        cls._validate_params(k=k)
        cls._validate_inputs(t=t)
        return _core.chi_square_cgf_scalar(float(t), float(k))

    @classmethod
    def _sample(cls, k: Union[int, float]) -> float:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        k : float
            Degrees of freedom (must be positive).

        Returns
        -------
        int
            One draw from ChiSquare(k), via the equivalent Gamma(k/2, 2).

        Raises
        ------
        TypeError
            If a parameter is of the wrong type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Seed the shared engine with `fastdist.seed` for reproducible draws.
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(k=k)
        return _core.chi_square_sample(float(k))
