# python/distributions/binomial.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Binomial:
    # Magic Methods
    __slots__ = ("n", "p")

    def __init__(self, n: int, p: int | float):
        Binomial._validate_params(n=n, p=p)
        self.n = int(n)
        self.p = float(p)

    def __repr__(self):
        return f"Binomial(n={self.n}, p={self.p})"

    @staticmethod
    def _validate_params(n: int, p: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(n, int):
            raise TypeError("n must be an integer")
        if not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if n < 0:
            raise ValueError("n must be a non-negative integer")
        if p < 0 or p > 1:
            raise ValueError("p must be in the interval [0, 1]")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        if x is not None and not isinstance(x, int):
            raise TypeError("x must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # Instance Methods
    def logpmf_scalar(self, x: int) -> float:
        self._validate_inputs(x=x)
        return _core.binomial_logpmf_scalar(x, self.n, self.p)

    def pmf_scalar(self, x: int) -> float:
        self._validate_inputs(x=x)
        return _core.binomial_pmf_scalar(x, self.n, self.p)

    def cdf_scalar(self, x: int) -> float:
        self._validate_inputs(x=x)
        return _core.binomial_cdf_scalar(x, self.n, self.p)

    def mean(self) -> float:
        return _core.binomial_mean(self.n, self.p)

    def variance(self) -> float:
        return _core.binomial_variance(self.n, self.p)

    def stddev(self) -> float:
        return _core.binomial_stddev(self.n, self.p)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.binomial_mgf_scalar(float(t), self.n, self.p)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.binomial_cgf_scalar(float(t), self.n, self.p)

    def sample(self) -> int:
        return _core.binomial_sample(self.n, self.p)

    # Static Methods
    @classmethod
    def _logpmf_scalar(cls, x: int, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(x=x)
        return _core.binomial_logpmf_scalar(x, n, float(p))

    @classmethod
    def _pmf_scalar(cls, x: int, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(x=x)
        return _core.binomial_pmf_scalar(x, n, float(p))

    @classmethod
    def _cdf_scalar(cls, x: int, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(x=x)
        return _core.binomial_cdf_scalar(x, n, float(p))

    @classmethod
    def _mean(cls, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        return _core.binomial_mean(n, float(p))

    @classmethod
    def _variance(cls, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        return _core.binomial_variance(n, float(p))

    @classmethod
    def _stddev(cls, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        return _core.binomial_stddev(n, float(p))

    @classmethod
    def _mgf_scalar(cls, t: int | float, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(t=t)
        return _core.binomial_mgf_scalar(float(t), n, float(p))

    @classmethod
    def _cgf_scalar(cls, t: int | float, n: int, p: int | float) -> float:
        cls._validate_params(n=n, p=p)
        cls._validate_inputs(t=t)
        return _core.binomial_cgf_scalar(float(t), n, float(p))

    @classmethod
    def _sample(cls, n: int, p: int | float) -> int:
        cls._validate_params(n=n, p=p)
        return _core.binomial_sample(n, float(p))
