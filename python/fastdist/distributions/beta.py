# python/fastdist/distributions/beta.py
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

class Beta:
    """
    Beta distribution.

    Represents a continuous random variable on the interval [0, 1], commonly
    used as a conjugate prior for a probability. Supports scalar evaluation of
    PDF, CDF, the moments, and sampling.

        f(x) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)

    Parameters
    ----------
    alpha : float
        First shape parameter (must be positive and finite).
    beta : float
        Second shape parameter (must be positive and finite).

    Attributes
    ----------
    alpha : float
        First shape parameter.
    beta : float
        Second shape parameter.

    Notes
    -----
    No MGF or CGF is provided. The Beta MGF is a confluent hypergeometric
    function with no elementary closed form, and the finite-sum approach that
    works for a bounded discrete support does not apply to a continuous one.

    The CDF is the regularized incomplete beta function, evaluated by a
    continued fraction, so it is accurate to roughly 1e-12 rather than to
    machine epsilon.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_alpha", "_beta")

    def __init__(self, alpha: Union[int, float], beta: Union[int, float]):
        self._validate_params(alpha=alpha, beta=beta)
        self._alpha = float(alpha)
        self._beta = float(beta)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._validate_params(alpha=value, beta=self._beta)
        self._alpha = float(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._validate_params(alpha=self._alpha, beta=value)
        self._beta = float(value)

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta})"

    @staticmethod
    def _validate_params(alpha: Union[int, float, None] = None, beta: Union[int, float, None] = None) -> None:
        """Internal validation shared by all methods."""
        if alpha is not None:
            if not isinstance(alpha, (int, float)):
                raise TypeError("alpha must be a real number")
            if not math.isfinite(alpha):
                raise ValueError("alpha must be finite")
            if alpha <= 0:
                raise ValueError("alpha must be positive")

        if beta is not None:
            if not isinstance(beta, (int, float)):
                raise TypeError("beta must be a real number")
            if not math.isfinite(beta):
                raise ValueError("beta must be finite")
            if beta <= 0:
                raise ValueError("beta must be positive")

    @staticmethod
    def _validate_inputs(x=None) -> None:
        """
        Validate the evaluation points passed to this distribution's methods.

        Parameters
        ----------
        x : float or None, optional
            Point at which to evaluate. Skipped when None.

        Raises
        ------
        TypeError
            If `x` is not a real number.

        Notes
        -----
        A value outside the support is not an error; it yields 0 or a
        boundary probability from the core, matching numpy's elementwise
        behaviour.
        """
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pdf_scalar(self, x: Union[int, float]) -> float:
        """
        Compute the Beta probability density function (PDF).

        Parameters
        ----------
        x : float
            Point at which to evaluate, in [0, 1].

        Returns
        -------
        float
            x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta).

        Raises
        ------
        TypeError
            If `x` is not a real number.
        """
        self._validate_inputs(x=x)
        return _core.beta_pdf_scalar(float(x), self.alpha, self.beta)

    def cdf_scalar(self, x: Union[int, float]) -> float:
        """
        Compute the Beta cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate, in [0, 1].

        Returns
        -------
        float
            the regularized incomplete beta I_x(alpha, beta).

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
        return _core.beta_cdf_scalar(float(x), self.alpha, self.beta)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            alpha / (alpha + beta).
        """
        return _core.beta_mean(self.alpha, self.beta)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)).
        """
        return _core.beta_variance(self.alpha, self.beta)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.beta_stddev(self.alpha, self.beta)

    def sample(self) -> float:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            Formed as A / (A + B) with A ~ Gamma(alpha, 1) and B ~ Gamma(beta, 1).

        Notes
        -----
        Seed the shared engine with `fastdist.seed` for reproducible draws.
        """
        return _core.beta_sample(self.alpha, self.beta)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], beta: Union[int, float]) -> float:
        """
        Compute the Beta probability density function (PDF).

        Parameters
        ----------
        x : float
            Point at which to evaluate, in [0, 1].
        alpha : float
            First shape parameter (must be positive).
        beta : float
            Second shape parameter (must be positive).

        Returns
        -------
        float
            x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta).

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
        cls._validate_params(alpha=alpha, beta=beta)
        cls._validate_inputs(x=x)
        return _core.beta_pdf_scalar(float(x), float(alpha), float(beta))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], beta: Union[int, float]) -> float:
        """
        Compute the Beta cumulative distribution function (CDF).

        Evaluates P(X <= x).

        Parameters
        ----------
        x : float
            Point at which to evaluate, in [0, 1].
        alpha : float
            First shape parameter (must be positive).
        beta : float
            Second shape parameter (must be positive).

        Returns
        -------
        float
            the regularized incomplete beta I_x(alpha, beta).

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
        cls._validate_params(alpha=alpha, beta=beta)
        cls._validate_inputs(x=x)
        return _core.beta_cdf_scalar(float(x), float(alpha), float(beta))

    @classmethod
    def _mean(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        alpha : float
            First shape parameter (must be positive).
        beta : float
            Second shape parameter (must be positive).

        Returns
        -------
        float
            alpha / (alpha + beta).

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
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_mean(float(alpha), float(beta))

    @classmethod
    def _variance(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        alpha : float
            First shape parameter (must be positive).
        beta : float
            Second shape parameter (must be positive).

        Returns
        -------
        float
            alpha*beta / ((alpha+beta)^2 * (alpha+beta+1)).

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
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_variance(float(alpha), float(beta))

    @classmethod
    def _stddev(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        alpha : float
            First shape parameter (must be positive).
        beta : float
            Second shape parameter (must be positive).

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
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_stddev(float(alpha), float(beta))

    @classmethod
    def _sample(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        alpha : float
            First shape parameter (must be positive).
        beta : float
            Second shape parameter (must be positive).

        Returns
        -------
        int
            Formed as A / (A + B) with A ~ Gamma(alpha, 1) and B ~ Gamma(beta, 1).

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
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_sample(float(alpha), float(beta))
