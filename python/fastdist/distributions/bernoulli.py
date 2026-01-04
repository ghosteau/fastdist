# python/distributions/bernoulli.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Bernoulli:
    # Magic Methods
    __slots__ = ("p",)

    def __init__(self, p: int | float):
        self._validate_params(p)
        self.p = float(p)

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    @staticmethod
    def _validate_params(p: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if p < 0 or p > 1:
            raise ValueError("p must be in the interval [0, 1]")

    @staticmethod
    def _validate_inputs(k=None, t=None) -> None:
        if k is not None and not isinstance(k, int):
            raise TypeError("k must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # Instance Methods
    def pmf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.bernoulli_pmf_scalar(k, self.p)

    def cdf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.bernoulli_cdf_scalar(k, self.p)

    def mean(self) -> float:
        return _core.bernoulli_mean(self.p)

    def variance(self) -> float:
        return _core.bernoulli_variance(self.p)

    def stddev(self) -> float:
        return _core.bernoulli_stddev(self.p)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.bernoulli_mgf(float(t), self.p)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.bernoulli_cgf(float(t), self.p)

    def sample(self) -> int:
        return _core.bernoulli_sample(self.p)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k: int, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.bernoulli_pmf_scalar(k, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.bernoulli_cdf_scalar(k, float(p))

    @classmethod
    def _mean(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.bernoulli_mean(float(p))

    @classmethod
    def _variance(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.bernoulli_variance(float(p))

    @classmethod
    def _stddev(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.bernoulli_stddev(float(p))

    @classmethod
    def _mgf_scalar(cls, t: int | float, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(t=t)
        return _core.bernoulli_mgf(float(t), float(p))

    @classmethod
    def _cgf_scalar(cls, t: int | float, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(t=t)
        return _core.bernoulli_cgf(float(t), float(p))

    @classmethod
    def _sample(cls, p: int | float) -> int:
        cls._validate_params(p=p)
        return _core.bernoulli_sample(float(p))
