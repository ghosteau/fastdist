# python/probforge/distributions/negative_binomial.py
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
from typing import Sequence, Union
from numpy.typing import NDArray

class NegativeBinomial:
    """
    Negative binomial distribution.

    Represents the number of failures observed before the `r`-th success in
    independent trials. Supports scalar evaluation of PMF, CDF, MGF, CGF, and
    sampling.

        P(X = k) = C(k+r-1, k) * p^r * (1-p)^k,  k = 0, 1, 2, ...

    Parameters
    ----------
    r : int
        Number of successes to wait for (must be a positive integer).
    p : float
        Probability of success on each trial (must be in (0, 1]).

    Attributes
    ----------
    r : int
        Number of successes to wait for.
    p : float
        Probability of success.

    Notes
    -----
    With ``r = 1`` this is the geometric distribution counted from 0, one less
    than the convention `Geometric` uses.

    At ``p = 1`` the distribution collapses onto ``k = 0``, which is accepted
    throughout and matches what SciPy reports.

    Often used in place of `Poisson` for overdispersed counts, since its
    variance exceeds its mean.

    This class is scalar-only; there is no batch or CUDA path for it yet.
    """

    # Magic Methods
    __slots__ = ("_r", "_p")

    def __init__(self, r: int, p: Union[int, float]):
        self._validate_params(r=r, p=p)
        self._r = r
        self._p = float(p)

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, value):
        self._validate_params(r=value)
        self._r = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._validate_params(p=value)
        self._p = float(value)

    def __repr__(self):
        return f"NegativeBinomial(r={self.r}, p={self.p})"

    @staticmethod
    def _validate_params(r: Union[int, None] = None, p: Union[int, float, None] = None) -> None:
        """Internal validation shared by all methods."""
        if r is not None:
            if not isinstance(r, int):
                raise TypeError("r must be an integer")
            if r <= 0:
                raise ValueError("r must be positive")
        if p is not None:
            if not isinstance(p, (int, float)):
                raise TypeError("p must be a real number")
            if not math.isfinite(p):
                raise ValueError("p must be finite")
            if not 0 <= p <= 1:
                raise ValueError("p must be in [0, 1]")

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
        behavior.
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
        Compute the NegativeBinomial probability mass function (PMF).

        Parameters
        ----------
        k : int
            Failure count at which to evaluate.

        Returns
        -------
        float
            C(k+r-1, k) * p^r * (1-p)^k.

        Raises
        ------
        TypeError
            If `k` is not an integer.
        """
        self._validate_inputs(k=k)
        return _core.negative_binomial_pmf_scalar(k, self.r, self.p)

    def cdf_scalar(self, k: int) -> float:
        """
        Compute the NegativeBinomial cumulative distribution function (CDF).

        Evaluates P(X <= k).

        Parameters
        ----------
        k : int
            Failure count at which to evaluate.

        Returns
        -------
        float
            the sum of the mass up to and including k.

        Raises
        ------
        TypeError
            If `k` is not an integer.
        """
        self._validate_inputs(k=k)
        return _core.negative_binomial_cdf_scalar(k, self.r, self.p)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            r * (1 - p) / p.
        """
        return _core.negative_binomial_mean(self.r, self.p)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            r * (1 - p) / p^2.
        """
        return _core.negative_binomial_variance(self.r, self.p)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of the variance.
        """
        return _core.negative_binomial_stddev(self.r, self.p)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the NegativeBinomial moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            (p / (1 - (1-p)*e^t))^r, defined while (1-p)*e^t < 1.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.negative_binomial_mgf_scalar(float(t), self.r, self.p)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        """
        Compute the NegativeBinomial cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.

        Returns
        -------
        float
            r * (log(p) - log(1 - (1-p)*e^t)), same domain as the MGF.

        Raises
        ------
        TypeError
            If `t` is not a real number.
        """
        self._validate_inputs(t=t)
        return _core.negative_binomial_cgf_scalar(float(t), self.r, self.p)

    def sample(self) -> int:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            One draw: the number of failures before the r-th success.

        Notes
        -----
        Seed the shared engine with `probforge.seed` for reproducible draws.
        """
        return _core.negative_binomial_sample(self.r, self.p)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, k: int, r: int, p: Union[int, float]) -> float:
        """
        Compute the NegativeBinomial probability mass function (PMF).

        Parameters
        ----------
        k : int
            Failure count at which to evaluate.
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            C(k+r-1, k) * p^r * (1-p)^k.

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
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(k=k)
        return _core.negative_binomial_pmf_scalar(k, r, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, r: int, p: Union[int, float]) -> float:
        """
        Compute the NegativeBinomial cumulative distribution function (CDF).

        Evaluates P(X <= k).

        Parameters
        ----------
        k : int
            Failure count at which to evaluate.
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            the sum of the mass up to and including k.

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
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(k=k)
        return _core.negative_binomial_cdf_scalar(k, r, float(p))

    @classmethod
    def _mean(cls, r: int, p: Union[int, float]) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            r * (1 - p) / p.

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
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_mean(r, float(p))

    @classmethod
    def _variance(cls, r: int, p: Union[int, float]) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            r * (1 - p) / p^2.

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
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_variance(r, float(p))

    @classmethod
    def _stddev(cls, r: int, p: Union[int, float]) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        r : int
            Number of successes to wait for (must be positive).
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
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_stddev(r, float(p))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], r: int, p: Union[int, float]) -> float:
        """
        Compute the NegativeBinomial moment generating function (MGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            (p / (1 - (1-p)*e^t))^r, defined while (1-p)*e^t < 1.

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
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(t=t)
        return _core.negative_binomial_mgf_scalar(float(t), r, float(p))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], r: int, p: Union[int, float]) -> float:
        """
        Compute the NegativeBinomial cumulant generating function (CGF).

        Parameters
        ----------
        t : float
            Point at which to evaluate the generating function.
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        float
            r * (log(p) - log(1 - (1-p)*e^t)), same domain as the MGF.

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
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(t=t)
        return _core.negative_binomial_cgf_scalar(float(t), r, float(p))

    @classmethod
    def _sample(cls, r: int, p: Union[int, float]) -> int:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        r : int
            Number of successes to wait for (must be positive).
        p : float
            Probability of success on each trial, in (0, 1].

        Returns
        -------
        int
            One draw: the number of failures before the r-th success.

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
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_sample(r, float(p))
