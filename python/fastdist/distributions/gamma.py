# python/fastdist/distributions/gamma.py

try:
    from .. import _fastdist as _core
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

class Gamma:
    """
    Gamma distribution.

    Represents a continuous positive random variable in the *shape-scale*
    parameterisation, generalising the exponential to a sum of `alpha`
    exponential waiting times. Supports scalar evaluation of the density, CDF,
    MGF, CGF, and sampling.

        f(x) = x^(alpha-1) * exp(-x/theta) / (Gamma(alpha) * theta^alpha),  x >= 0

    Parameters
    ----------
    alpha : float
        Shape parameter (must be positive and finite).
    theta : float
        Scale parameter (must be positive and finite). Note this is a scale,
        not a rate; the rate parameterisation uses ``1 / theta``.

    Attributes
    ----------
    alpha : float
        Shape parameter.
    theta : float
        Scale parameter.

    Notes
    -----
    `Exponential` is the special case ``alpha = 1``, and `ChiSquare` with `k`
    degrees of freedom is ``alpha = k/2, theta = 2``.

    The instance method is named `pmf_scalar` for historical reasons but
    returns a probability *density*: this distribution is continuous, so there
    is no mass at a point. The classmethod form is named `_pdf_scalar`
    correctly.

    The CDF is the regularized lower incomplete gamma function, evaluated
    iteratively, so it is accurate to roughly 1e-12 rather than to machine
    epsilon.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_alpha", "_theta")

    def __init__(self, alpha: Union[int, float], theta: Union[int, float]):
        self._validate_params(alpha=alpha, theta=theta)
        self._alpha = float(alpha)
        self._theta = float(theta)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._validate_params(alpha=value)
        self._alpha = float(value)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._validate_params(theta=value)
        self._theta = float(value)

    def __repr__(self):
        return f"Gamma(alpha={self.alpha}, theta={self.theta})"

    @staticmethod
    def _validate_params(alpha: Union[int, float, None] = None, theta: Union[int, float, None] = None) -> None:
        """Internal validation shared by all methods."""
        if alpha is not None:
            if not isinstance(alpha, (int, float)):
                raise TypeError("alpha must be a real number")
            if not math.isfinite(alpha):
                raise ValueError("alpha must be finite")
            if alpha <= 0:
                raise ValueError("alpha must be positive")
        if theta is not None:
            if not isinstance(theta, (int, float)):
                raise TypeError("theta must be a real number")
            if not math.isfinite(theta):
                raise ValueError("theta must be finite")
            if theta <= 0:
                raise ValueError("theta must be positive")

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
    def pmf_scalar(self, x: Union[int, float]) -> float:
        """
        Compute the Gamma probability mass function (PMF).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.

        Returns
        -------
        float
            The computed value.

        Raises
        ------
        TypeError
            If `x` is not a real number.
        """
        self._validate_inputs(x=x)
        return _core.gamma_pdf_scalar(x, self.alpha, self.theta)

    def cdf_scalar(self, x: Union[int, float]) -> float:
        """
        Compute the Gamma cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.

        Returns
        -------
        float
            the regularized lower incomplete gamma P(alpha, x/theta).

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
        return _core.gamma_cdf_scalar(x, self.alpha, self.theta)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            alpha * theta.
        """
        return _core.gamma_mean(self.alpha, self.theta)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            alpha * theta^2.
        """
        return _core.gamma_variance(self.alpha, self.theta)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.gamma_stddev(self.alpha, self.theta)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the Gamma moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            (1 - theta*t)^(-alpha), defined for t < 1/theta.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.gamma_mgf_scalar(t, self.alpha, self.theta)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the Gamma cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            -alpha * log(1 - theta*t), defined for t < 1/theta.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.gamma_cgf_scalar(t, self.alpha, self.theta)

    def sample(self) -> float:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            One draw from Gamma(alpha, theta).

        Notes
        -----
        Seed the shared engine with `fastdist.seed` for reproducible draws.
        """
        return _core.gamma_sample(self.alpha, self.theta)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the Gamma probability density function (PDF).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        float
            x^(alpha-1) * exp(-x/theta) / (Gamma(alpha) * theta^alpha).

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
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(x=x)
        return _core.gamma_pdf_scalar(float(x), float(alpha), float(theta))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the Gamma cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate; non-negative.
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        float
            the regularized lower incomplete gamma P(alpha, x/theta).

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
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(x=x)
        return _core.gamma_cdf_scalar(float(x), float(alpha), float(theta))

    @classmethod
    def _mean(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        float
            alpha * theta.

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
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_mean(float(alpha), float(theta))

    @classmethod
    def _variance(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        float
            alpha * theta^2.

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
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_variance(float(alpha), float(theta))

    @classmethod
    def _stddev(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

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
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_stddev(float(alpha), float(theta))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the Gamma moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        float
            (1 - theta*t)^(-alpha), defined for t < 1/theta.

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
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(t=t)
        return _core.gamma_mgf_scalar(float(t), float(alpha), float(theta))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Compute the Gamma cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        float
            -alpha * log(1 - theta*t), defined for t < 1/theta.

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
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(t=t)
        return _core.gamma_cgf_scalar(float(t), float(alpha), float(theta))

    @classmethod
    def _sample(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        alpha : float
            Shape parameter (must be positive).
        theta : float
            Scale parameter (must be positive).

        Returns
        -------
        int
            One draw from Gamma(alpha, theta).

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
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_sample(float(alpha), float(theta))
