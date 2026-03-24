# python/distributions/poisson.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

from . import Real, Sequence, Union, NDArray

class NegativeBinomial:
    # Magic Methods
    __slots__ = ("_r", "_p")

    def __init__(self, r: int, p: Union[int, float]):
        self._validate_params(r=r, p=p)
        self._r = r
        self._p = float(p)

    @property
    def r(self):
        return self._r

    @property
    def p(self):
        return self._p

    @r.setter
    def r(self, value):
        self._validate_params(r=value)
        self._r = value

    @p.setter
    def p(self, value):
        self._validate_params(p=value)
        self._p = float(value)

    def __repr__(self):
        return f"NegativeBinomial(r={self.r}, p={self.p})"

    @staticmethod
    def _validate_params(r: int = None, p: Union[int, float] = None) -> None:
        """Internal validation shared by all methods."""
        if r is not None:
            if not isinstance(r, int):
                raise TypeError("r must be an integer")
            if r <= 0:
                raise ValueError("r must be positive")
        if p is not None:
            if not isinstance(p, (int, float)):
                raise TypeError("p must be a real number")
            if not 0 <= p <= 1:
                raise ValueError("p must be in [0, 1]")

    @staticmethod
    def _validate_inputs(k=None, t=None) -> None:
        if k is not None and not isinstance(k, int):
            raise TypeError("k must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pmf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.negative_binomial_pmf_scalar(k, self.r, self.p)

    def cdf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.negative_binomial_cdf_scalar(k, self.r, self.p)

    def mean(self) -> float:
        return _core.negative_binomial_mean(self.r, self.p)

    def variance(self) -> float:
        return _core.negative_binomial_variance(self.r, self.p)

    def stddev(self) -> float:
        return _core.negative_binomial_stddev(self.r, self.p)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.negative_binomial_mgf_scalar(float(t), self.r, self.p)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.negative_binomial_cgf_scalar(float(t), self.r, self.p)

    def sample(self) -> int:
        return _core.negative_binomial_sample(self.r, self.p)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, k: int, r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(k=k)
        return _core.negative_binomial_pmf_scalar(k, r, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(k=k)
        return _core.negative_binomial_cdf_scalar(k, r, float(p))

    @classmethod
    def _mean(cls, r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_mean(r, float(p))

    @classmethod
    def _variance(cls, r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_variance(r, float(p))

    @classmethod
    def _stddev(cls, r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_stddev(r, float(p))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(t=t)
        return _core.negative_binomial_mgf_scalar(float(t), r, float(p))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], r: int, p: Union[int, float]) -> float:
        cls._validate_params(r=r, p=p)
        cls._validate_inputs(t=t)
        return _core.negative_binomial_cgf_scalar(float(t), r, float(p))

    @classmethod
    def _sample(cls, r: int, p: Union[int, float]) -> int:
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_sample(r, float(p))
