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

from typing import Union


class Hypergeometric:
    """
    Successes in a sample drawn without replacement.

    This is the binomial's counterpart for sampling from a finite pool: each
    draw removes an item, so the draws are not independent and the variance is
    smaller than a binomial's by the finite population correction. It is the
    exact null distribution behind Fisher's exact test and behind the
    enrichment tests used in feature selection.

        population  N  size of the whole population
        successes   K  items in the population that count as a success
        draws       n  items drawn, without replacement
        k              successes observed among the draws

        P(X = k) = C(K, k) C(N - K, n - k) / C(N, n)

    The support is max(0, n - (N - K)) <= k <= min(n, K).

    The parameters are named rather than positional-by-convention because
    SciPy's (M, n, N) ordering is easy to transpose silently; SciPy's M is this
    class's ``population``, its n is ``successes`` and its N is ``draws``.

    >>> # 20 items, 7 of them defective, draw 5: chance of exactly 2 defective
    >>> Hypergeometric(population=20, successes=7, draws=5).pmf_scalar(2)
    0.3873839009287926
    """

    __slots__ = ("_population", "_successes", "_draws")

    def __init__(self, population: int, successes: int, draws: int):
        Hypergeometric._validate_params(population=population, successes=successes, draws=draws)
        self._population = int(population)
        self._successes = int(successes)
        self._draws = int(draws)

    # ------------------------------------------------------------------------
    # Properties
    #
    # Each setter revalidates the whole triple rather than its own value alone,
    # because the constraints are between the parameters: successes and draws
    # must both fit inside the population.
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------
    @staticmethod
    def _validate_params(population: Union[int, None] = None,
                         successes: Union[int, None] = None,
                         draws: Union[int, None] = None) -> None:
        """Internal validation shared by all methods."""
        for name, value in (("population", population), ("successes", successes), ("draws", draws)):
            if value is None:
                continue
            # bool is a subclass of int; a True population is a mistake, not a 1.
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if population is not None and successes is not None and successes > population:
            raise ValueError("successes must not exceed population")
        if population is not None and draws is not None and draws > population:
            raise ValueError("draws must not exceed population")

    @staticmethod
    def _validate_inputs(k=None, t=None) -> None:
        if k is not None and (not isinstance(k, int) or isinstance(k, bool)):
            raise TypeError("k must be an integer")
        if t is not None and (not isinstance(t, (int, float)) or isinstance(t, bool)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------
    # Support
    # ------------------------------------------------------------------------
    def support(self) -> "tuple[int, int]":
        """
        The inclusive range of k with non-zero probability.

        A draw cannot hold more successes than exist or than were drawn, and if
        it is larger than the pool of failures the surplus must be successes.
        """
        low = max(0, self._draws - (self._population - self._successes))
        high = min(self._draws, self._successes)
        return low, high

    # ------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------
    def logpmf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.hypergeometric_logpmf_scalar(
            k=k, population=self.population, successes=self.successes, draws=self.draws)

    def pmf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.hypergeometric_pmf_scalar(
            k=k, population=self.population, successes=self.successes, draws=self.draws)

    def cdf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.hypergeometric_cdf_scalar(
            k=k, population=self.population, successes=self.successes, draws=self.draws)

    def mean(self) -> float:
        return _core.hypergeometric_mean(
            population=self.population, successes=self.successes, draws=self.draws)

    def variance(self) -> float:
        return _core.hypergeometric_variance(
            population=self.population, successes=self.successes, draws=self.draws)

    def stddev(self) -> float:
        return _core.hypergeometric_stddev(
            population=self.population, successes=self.successes, draws=self.draws)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.hypergeometric_mgf_scalar(
            t=float(t), population=self.population, successes=self.successes, draws=self.draws)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.hypergeometric_cgf_scalar(
            t=float(t), population=self.population, successes=self.successes, draws=self.draws)

    def sample(self) -> int:
        return _core.hypergeometric_sample(
            population=self.population, successes=self.successes, draws=self.draws)

    # ------------------------------------------------------------------------
    # Scalar Static Methods
    #
    # The same calls without building an instance, for one-off evaluation.
    # ------------------------------------------------------------------------
    @classmethod
    def _logpmf_scalar(cls, k: int, population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(k=k)
        return _core.hypergeometric_logpmf_scalar(
            k=k, population=population, successes=successes, draws=draws)

    @classmethod
    def _pmf_scalar(cls, k: int, population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(k=k)
        return _core.hypergeometric_pmf_scalar(
            k=k, population=population, successes=successes, draws=draws)

    @classmethod
    def _cdf_scalar(cls, k: int, population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(k=k)
        return _core.hypergeometric_cdf_scalar(
            k=k, population=population, successes=successes, draws=draws)

    @classmethod
    def _mean(cls, population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_mean(population=population, successes=successes, draws=draws)

    @classmethod
    def _variance(cls, population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_variance(population=population, successes=successes, draws=draws)

    @classmethod
    def _stddev(cls, population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_stddev(population=population, successes=successes, draws=draws)

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(t=t)
        return _core.hypergeometric_mgf_scalar(
            t=float(t), population=population, successes=successes, draws=draws)

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], population: int, successes: int, draws: int) -> float:
        cls._validate_params(population=population, successes=successes, draws=draws)
        cls._validate_inputs(t=t)
        return _core.hypergeometric_cgf_scalar(
            t=float(t), population=population, successes=successes, draws=draws)

    @classmethod
    def _sample(cls, population: int, successes: int, draws: int) -> int:
        cls._validate_params(population=population, successes=successes, draws=draws)
        return _core.hypergeometric_sample(population=population, successes=successes, draws=draws)
