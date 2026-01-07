# python/distributions/geometric.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Geometric:
    # Magic Methods
    __slots__ = ("_p",)

    def __init__(self, p: int | float):
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
    def _validate_params(p: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if not (0 < p <= 1):
            raise ValueError("p must be in the interval (0, 1]")

    @staticmethod
    def _validate_inputs(k=None, t=None) -> None:
        if k is not None and not isinstance(k, int):
            raise TypeError("k must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # Instance Methods
    def pmf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.geometric_pmf_scalar(k, self.p)

    def cdf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.geometric_cdf_scalar(k, self.p)

    def mean(self) -> float:
        return _core.geometric_mean(self.p)

    def variance(self) -> float:
        return _core.geometric_variance(self.p)

    def stddev(self) -> float:
        return _core.geometric_stddev(self.p)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.geometric_mgf_scalar(float(t), self.p)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.geometric_cgf_scalar(float(t), self.p)

    def sample(self) -> int:
        return _core.geometric_sample(self.p)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k: int, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.geometric_pmf_scalar(k, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.geometric_cdf_scalar(k, float(p))

    @classmethod
    def _mean(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.geometric_mean(float(p))

    @classmethod
    def _variance(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.geometric_variance(float(p))

    @classmethod
    def _stddev(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.geometric_stddev(float(p))

    @classmethod
    def _mgf_scalar(cls, t: int | float, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.geometric_mgf_scalar(t, float(p))

    @classmethod
    def _cgf_scalar(cls, t: int | float, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.geometric_cgf_scalar(t, float(p))

    @classmethod
    def _sample(cls, p: int | float) -> int:
        cls._validate_params(p=p)
        return _core.geometric_sample(float(p))
