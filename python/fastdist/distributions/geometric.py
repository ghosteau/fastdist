# python/fastdist/distributions/geometric.py
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

class Geometric:
    """
    Geometric distribution.

    Represents the number of independent trials up to and including the first
    success. Supports scalar evaluation of PMF, CDF, MGF, CGF, and sampling.

        P(X = k) = p * (1-p)^(k-1),  k = 1, 2, 3, ...

    Parameters
    ----------
    p : float
        Probability of success on each trial (must be in (0, 1]).

    Attributes
    ----------
    p : float
        Probability of success.

    Notes
    -----
    This class counts *trials*, so the support starts at 1 and the mean is
    ``1 / p``. The other common convention counts failures before the first
    success and starts at 0; that is `NegativeBinomial` with ``r = 1``.

    This distribution is memoryless, the only discrete distribution that is.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_p",)

    def __init__(self, p: Union[int, float]):
        self._validate_params(p=p)
        self._p = float(p)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._validate_params(p=value)
        self._p = float(value)

    def __repr__(self):
        return f"Geometric(p={self.p})"

    @staticmethod
    def _validate_params(p: Union[int, float]) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if not math.isfinite(p):
            raise ValueError("p must be finite")
        if not (0 < p <= 1):
            raise ValueError("p must be in the interval (0, 1]")

    @staticmethod
    def _validate_inputs(k=None, t=None) -> None:
        """
        Validate the evaluation points passed to this distribution's methods.

        Parameters
        ----------
        k : int or None, optional
            Point at which to evaluate. Skipped when None.
        t : float or None, optional
            Point at which to evaluate. Skipped when None.

        Raises
        ------
        TypeError
            If `k` is not an integer. If `t` is not a real number.

        Notes
        -----
        A value outside the support is not an error; it yields 0 or a
        boundary probability from the core, matching numpy's elementwise
        behaviour.
        """
        if k is not None and not isinstance(k, int):
            raise TypeError("k must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pmf_scalar(self, k: int) -> float:
        """
        Compute the Geometric probability mass function (PMF).

        Parameters
        ----------
        k : int
            Trial index at which to evaluate; the support starts at 1.

        Returns
        -------
        float
            p * (1-p)^(k-1).

        Raises
        ------
        TypeError
            If `k` is not an integer.
        """
        self._validate_inputs(k=k)
        return _core.geometric_pmf_scalar(k, self.p)

    def cdf_scalar(self, k: int) -> float:
        """
        Compute the Geometric cumulative distribution function (CDF).

        Evaluates P(X <= k).

        Parameters
        ----------
        k : int
            Trial index at which to evaluate; the support starts at 1.

        Returns
        -------
        float
            1 - (1-p)^k.

        Raises
        ------
        TypeError
            If `k` is not an integer.
        """
        self._validate_inputs(k=k)
        return _core.geometric_cdf_scalar(k, self.p)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            1 / p.
        """
        return _core.geometric_mean(self.p)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            (1 - p) / p^2.
        """
        return _core.geometric_variance(self.p)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.geometric_stddev(self.p)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the Geometric moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            p*e^t / (1 - (1-p)*e^t), defined for t < -log(1-p).

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.geometric_mgf_scalar(float(t), self.p)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the Geometric cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            log(p) + t - log(1 - (1-p)*e^t), same domain as the MGF.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.geometric_cgf_scalar(float(t), self.p)

    def sample(self) -> int:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            One draw: the index of the first success, so at least 1.

        Notes
        -----
        Seed the shared engine with `fastdist.seed` for reproducible draws.
        """
        return _core.geometric_sample(self.p)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, k: int, p: Union[int, float]) -> float:
        """
        Compute the Geometric probability mass function (PMF).

        Parameters
        ----------
        k : int
            Trial index at which to evaluate; the support starts at 1.
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            p * (1-p)^(k-1).

        Raises
        ------
        TypeError
            If `k` is not an integer. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.geometric_pmf_scalar(k, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, p: Union[int, float]) -> float:
        """
        Compute the Geometric cumulative distribution function (CDF).

        Evaluates P(X <= k).

        Parameters
        ----------
        k : int
            Trial index at which to evaluate; the support starts at 1.
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            1 - (1-p)^k.

        Raises
        ------
        TypeError
            If `k` is not an integer. If a parameter is of the wrong
            type.
        ValueError
            If a parameter is outside its valid range.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.geometric_cdf_scalar(k, float(p))

    @classmethod
    def _mean(cls, p: Union[int, float]) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            1 / p.

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
        cls._validate_params(p=p)
        return _core.geometric_mean(float(p))

    @classmethod
    def _variance(cls, p: Union[int, float]) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            (1 - p) / p^2.

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
        cls._validate_params(p=p)
        return _core.geometric_variance(float(p))

    @classmethod
    def _stddev(cls, p: Union[int, float]) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        p : float
            Probability of success on each trial, in (0, 1].

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
        cls._validate_params(p=p)
        return _core.geometric_stddev(float(p))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], p: Union[int, float]) -> float:
        """
        Compute the Geometric moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            p*e^t / (1 - (1-p)*e^t), defined for t < -log(1-p).

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
        cls._validate_params(p=p)
        return _core.geometric_mgf_scalar(t, float(p))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], p: Union[int, float]) -> float:
        """
        Compute the Geometric cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            log(p) + t - log(1 - (1-p)*e^t), same domain as the MGF.

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
        cls._validate_params(p=p)
        return _core.geometric_cgf_scalar(t, float(p))

    @classmethod
    def _sample(cls, p: Union[int, float]) -> int:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        int
            One draw: the index of the first success, so at least 1.

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
        cls._validate_params(p=p)
        return _core.geometric_sample(float(p))
