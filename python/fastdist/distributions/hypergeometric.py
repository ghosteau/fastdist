# python/fastdist/distributions/hypergeometric.py
try:
    from fastdist import _fastdist as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "fastdist's compiled extension (_fastdist) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

from numbers import Integral, Real
from typing import Tuple, Union


class Hypergeometric:
    """
    Hypergeometric distribution.

    Represents the number of successes in a sample drawn *without* replacement
    from a finite population, so unlike `Binomial` the trials are not
    independent and the variance is smaller by the finite population
    correction. This is the exact null distribution behind Fisher's exact test.
    Supports scalar evaluation of the PMF, log PMF, CDF, MGF, CGF, and sampling.

        P(X = k) = C(K, k) * C(N - K, n - k) / C(N, n)

    Parameters
    ----------
    population : int
        Size of the whole population, N (must be non-negative).
    successes : int
        Items counting as a success, K (must be in [0, population]).
    draws : int
        Items drawn without replacement, n (must be in [0, population]).

    Attributes
    ----------
    population : int
        Size of the whole population.
    successes : int
        Items counting as a success.
    draws : int
        Items drawn without replacement.

    Notes
    -----
    Named parameters rather than SciPy's ``hypergeom(M, n, N)``, whose order is
    easy to transpose; its `M`, `n`, `N` are `population`, `successes`, `draws`.

    The support does not always start at zero -- see `support` rather than
    assuming a lower bound of 0. Scalar-only; no batch or CUDA path yet.

    Examples
    --------
    >>> # 20 items, 7 of them defective; draw 5 without replacement
    >>> h = Hypergeometric(population=20, successes=7, draws=5)
    >>> round(h.pmf_scalar(2), 12)
    0.387383900929
    >>> h.mean()
    1.75
    >>> h.support()
    (0, 5)
    """

    # Magic Methods
    __slots__ = ("_population", "_successes", "_draws")

    def __init__(self, population: int, successes: int, draws: int):
        """
        Initialize a Hypergeometric distribution instance.

        Parameters
        ----------
        population : int
            Size of the whole population (must be non-negative).
        successes : int
            Items in the population that count as a success.
        draws : int
            Items drawn without replacement.

        Raises
        ------
        TypeError
            If any parameter is not an integer.
        ValueError
            If any parameter is negative, or if `successes` or `draws` exceeds
            `population`.
        """

        Hypergeometric._validate_params(population=population, successes=successes, draws=draws)
        self._population = int(population)
        self._successes = int(successes)
        self._draws = int(draws)

    @property
    def population(self) -> int:
        return self._population

    @population.setter
    def population(self, value: int) -> None:
        self._validate_params(population=value, successes=self._successes, draws=self._draws)
        self._population = int(value)

    @property
    def successes(self) -> int:
        return self._successes

    @successes.setter
    def successes(self, value: int) -> None:
        self._validate_params(population=self._population, successes=value, draws=self._draws)
        self._successes = int(value)

    @property
    def draws(self) -> int:
        return self._draws

    @draws.setter
    def draws(self, value: int) -> None:
        self._validate_params(population=self._population, successes=self._successes, draws=value)
        self._draws = int(value)

    def __repr__(self) -> str:
        return (f"Hypergeometric(population={self.population}, "
                f"successes={self.successes}, draws={self.draws})")

    @staticmethod
    def _validate_params(population: Union[int, None] = None,
                         successes: Union[int, None] = None,
                         draws: Union[int, None] = None) -> None:
        """
        Validate the population, success and draw counts.

        Parameters
        ----------
        population : int or None, optional
            Size of the whole population. Skipped when None.
        successes : int or None, optional
            Items in the population that count as a success. Skipped when None.
        draws : int or None, optional
            Items drawn without replacement. Skipped when None.

        Raises
        ------
        TypeError
            If a supplied parameter is not an integer.
        ValueError
            If a supplied parameter is negative, or if `successes` or `draws`
            exceeds `population`.

        Notes
        -----
        The constraints run between the parameters, not just over each one, so
        the property setters pass the whole triple rather than their own value
        alone. `bool` is rejected despite being a subclass of `int`: a `True`
        population is a mistake, not a 1.
        """

        for name, value in (("population", population), ("successes", successes), ("draws", draws)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if population is not None and successes is not None and successes > population:
            raise ValueError("successes must not exceed population")
        if population is not None and draws is not None and draws > population:
            raise ValueError("draws must not exceed population")

    @staticmethod
    def _validate_inputs(k: Union[int, None] = None, t: Union[Real, None] = None) -> None:
        """
        Validate inputs for Hypergeometric methods.

        Parameters
        ----------
        k : int or None, optional
            Observed success count at which to evaluate. Skipped when None.
        t : float or None, optional
            Point at which to evaluate a generating function. Skipped when None.

        Raises
        ------
        TypeError
            If `k` is not an integer or `t` is not a real number.

        Notes
        -----
        A `k` outside the support is not an error; it yields 0 from the PMF and
        0 or 1 from the CDF, matching the C++ core.
        """

        if k is not None and (isinstance(k, bool) or not isinstance(k, Integral)):
            raise TypeError("k must be an integer")
        if t is not None and (isinstance(t, bool) or not isinstance(t, Real)):
            raise TypeError("t must be a real number")

    def support(self) -> Tuple[int, int]:
        """
        Return the inclusive range of `k` with non-zero probability.

        Returns
        -------
        tuple of (int, int)
            Lowest and highest attainable success count.

        Notes
        -----
        The lower bound is not always zero. A draw cannot hold more successes
        than exist or than were drawn, and if it is larger than the pool of
        failures the surplus must be successes, so drawing 9 of 10 items from a
        population holding 9 successes must turn up at least 8.
        """

        low = max(0, self._draws - (self._population - self._successes))
        high = min(self._draws, self._successes)
        return low, high

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def logpmf_scalar(self, k: int) -> float:
        """
        Compute the Hypergeometric log probability mass function (log PMF).

        Parameters
        ----------
        k : int
            Success count at which to evaluate.

        Returns
        -------
        float
            Natural log of the probability mass at `k`, or ``-inf`` outside the
            support.

        Raises
        ------
        TypeError
            If `k` is not an integer.

        Notes
        -----
        Preferred over ``log(pmf_scalar(k))`` for large populations: the
        coefficients are evaluated through `lgamma`, so the factorials inside
        them cannot overflow.
        """

        self._validate_inputs(k=k)
        return _core.hypergeometric_logpmf_scalar(
            k=k, population=self.population, successes=self.successes, draws=self.draws)

    def pmf_scalar(self, k: int) -> float:
        """
        Compute the Hypergeometric probability mass function (PMF).

        Evaluates P(X = k), the probability of drawing exactly `k` successes.

        Parameters
        ----------
        k : int
            Success count at which to evaluate.

        Returns
        -------
        float
            Probability mass at `k`, or 0.0 outside the support.

        Raises
        ------
        TypeError
            If `k` is not an integer.
        """

        self._validate_inputs(k=k)
        return _core.hypergeometric_pmf_scalar(
            k=k, population=self.population, successes=self.successes, draws=self.draws)

    def cdf_scalar(self, k: int) -> float:
        """
        Compute the Hypergeometric cumulative distribution function (CDF).

        Evaluates P(X <= k) by summing the mass from the bottom of the support.

        Parameters
        ----------
        k : int
            Success count at which to evaluate.

        Returns
        -------
        float
            Cumulative probability at `k`, clamped to [0, 1]. Returns 0.0 below
            the support and 1.0 at or above its top.

        Raises
        ------
        TypeError
            If `k` is not an integer.

        Notes
        -----
        The upper tail ``1 - cdf_scalar(a - 1)`` is the one-sided p-value of
        Fisher's exact test for a 2x2 table.
        """

        self._validate_inputs(k=k)
        return _core.hypergeometric_cdf_scalar(
            k=k, population=self.population, successes=self.successes, draws=self.draws)

    def mean(self) -> float:
        """
        Compute the mean of the distribution.

        Returns
        -------
        float
            ``draws * successes / population``, or nan if `population` is 0.
        """

        return _core.hypergeometric_mean(
            population=self.population, successes=self.successes, draws=self.draws)

    def variance(self) -> float:
        """
        Compute the variance of the distribution.

        Returns
        -------
        float
            ``n * (K/N) * ((N-K)/N) * ((N-n)/(N-1))``, or nan if `population`
            is 0.

        Notes
        -----
        The last factor is the finite population correction, and it is what
        separates this from a binomial variance: it falls to zero when the
        whole population is drawn. A `population` of 1 returns 0.0, where the
        draw is deterministic and the correction would evaluate 0/0; SciPy
        reports nan for that case.
        """

        return _core.hypergeometric_variance(
            population=self.population, successes=self.successes, draws=self.draws)

    def stddev(self) -> float:
        """
        Compute the standard deviation of the distribution.

        Returns
        -------
        float
            Square root of `variance`, or nan if `population` is 0.
        """

        return _core.hypergeometric_stddev(
            population=self.population, successes=self.successes, draws=self.draws)

    def mgf_scalar(self, t: Real) -> float:
        """
        Compute the moment generating function (MGF) at `t`.

        Parameters
        ----------
        t : float
            Point at which to evaluate the MGF.

        Returns
        -------
        float
            ``sum_k exp(t * k) * P(X = k)``, or nan if `population` is 0. May
            be ``inf`` for large `t`; use `cgf_scalar` there.

        Raises
        ------
        TypeError
            If `t` is not a real number.

        Notes
        -----
        The Hypergeometric MGF has no elementary closed form -- it is a
        Gaussian hypergeometric 2F1 -- but the support is finite, so the
        defining sum above is exact rather than an approximation.
        """

        self._validate_inputs(t=t)
        return _core.hypergeometric_mgf_scalar(
            t=float(t), population=self.population, successes=self.successes, draws=self.draws)

    def cgf_scalar(self, t: Real) -> float:
        """
        Compute the cumulant generating function (CGF) at `t`.

        Parameters
        ----------
        t : float
            Point at which to evaluate the CGF.

        Returns
        -------
        float
            ``log(mgf_scalar(t))``, or nan if `population` is 0.

        Raises
        ------
        TypeError
            If `t` is not a real number.

        Notes
        -----
        Accumulated by log-sum-exp, so this stays finite well past the point
        where `mgf_scalar` overflows to ``inf``.
        """

        self._validate_inputs(t=t)
        return _core.hypergeometric_cgf_scalar(
            t=float(t), population=self.population, successes=self.successes, draws=self.draws)

    def sample(self) -> int:
        """
        Draw a random sample from the distribution.

        Returns
        -------
        int
            Successes observed in `draws` items taken without replacement.

        Notes
        -----
        There is no ``std::hypergeometric_distribution``, so the draw is
        simulated directly: the chance that the next item is a success is the
        proportion of successes among what is left. That is exact for any
        parameters rather than an approximation that degrades at the edges.

        Seed the shared engine with `fastdist.seed` for reproducible draws.
        """

        return _core.hypergeometric_sample(
            population=self.population, successes=self.successes, draws=self.draws)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _logpmf_scalar(cls, k: int, population: int, successes: int, draws: int) -> float:
        """
        Compute the log PMF without constructing an instance.

        Parameters
        ----------
        k : int
            Success count at which to evaluate.
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            Natural log of the probability mass at `k`.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(k=k)
        return _core.hypergeometric_logpmf_scalar(
            k=k, population=population, successes=successes, draws=draws)

    @classmethod
    def _pmf_scalar(cls, k: int, population: int, successes: int, draws: int) -> float:
        """
        Compute the PMF without constructing an instance.

        Parameters
        ----------
        k : int
            Success count at which to evaluate.
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            Probability mass at `k`.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(k=k)
        return _core.hypergeometric_pmf_scalar(
            k=k, population=population, successes=successes, draws=draws)

    @classmethod
    def _cdf_scalar(cls, k: int, population: int, successes: int, draws: int) -> float:
        """
        Compute the CDF without constructing an instance.

        Parameters
        ----------
        k : int
            Success count at which to evaluate.
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            Cumulative probability at `k`.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(k=k)
        return _core.hypergeometric_cdf_scalar(
            k=k, population=population, successes=successes, draws=draws)

    @classmethod
    def _mean(cls, population: int, successes: int, draws: int) -> float:
        """
        Compute the mean without constructing an instance.

        Parameters
        ----------
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            ``draws * successes / population``.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_mean(population=population, successes=successes, draws=draws)

    @classmethod
    def _variance(cls, population: int, successes: int, draws: int) -> float:
        """
        Compute the variance without constructing an instance.

        Parameters
        ----------
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            Variance, including the finite population correction.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_variance(population=population, successes=successes, draws=draws)

    @classmethod
    def _stddev(cls, population: int, successes: int, draws: int) -> float:
        """
        Compute the standard deviation without constructing an instance.

        Parameters
        ----------
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            Square root of the variance.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_stddev(population=population, successes=successes, draws=draws)

    @classmethod
    def _mgf_scalar(cls, t: Real, population: int, successes: int, draws: int) -> float:
        """
        Compute the MGF without constructing an instance.

        Parameters
        ----------
        t : float
            Point at which to evaluate the MGF.
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            MGF at `t`.

        Raises
        ------
        TypeError
            If `t` is not a real number or a parameter is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(t=t)
        return _core.hypergeometric_mgf_scalar(
            t=float(t), population=population, successes=successes, draws=draws)

    @classmethod
    def _cgf_scalar(cls, t: Real, population: int, successes: int, draws: int) -> float:
        """
        Compute the CGF without constructing an instance.

        Parameters
        ----------
        t : float
            Point at which to evaluate the CGF.
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        float
            CGF at `t`.

        Raises
        ------
        TypeError
            If `t` is not a real number or a parameter is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(t=t)
        return _core.hypergeometric_cgf_scalar(
            t=float(t), population=population, successes=successes, draws=draws)

    @classmethod
    def _sample(cls, population: int, successes: int, draws: int) -> int:
        """
        Draw a sample without constructing an instance.

        Parameters
        ----------
        population, successes, draws : int
            Distribution parameters, validated on every call.

        Returns
        -------
        int
            Successes observed in `draws` items taken without replacement.

        Raises
        ------
        TypeError
            If any argument is not an integer.
        ValueError
            If the parameters cannot describe a draw.
        """

        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_sample(population=population, successes=successes, draws=draws)
