# python/distributions/poisson.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Poisson:
    # Magic Methods
    __slots__ = "lambda_"

    def __init__(self, lambda_: int | float):
        Poisson._validate_params(lambda_=lambda_)
        self.lambda_ = float(lambda_)

    def __repr__(self):
        return f"Poisson(lambda_={self.lambda_})"

    @staticmethod
    def _validate_params(lambda_: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(lambda_, (int, float)):
            raise TypeError("lambda_ must be a real number")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # Instance Methods
    def pmf_scalar(self, x: int | float) -> float:
        self._validate_inputs(x=x)
        return _core.poisson_pmf_scalar(float(x), self.lambda_)

    def cdf_scalar(self, x: int | float) -> float:
        self._validate_inputs(x=x)
        return _core.poisson_cdf_scalar(float(x), self.lambda_)

    def mean(self) -> float:
        return _core.poisson_mean(self.lambda_)

    def variance(self) -> float:
        return _core.poisson_variance(self.lambda_)

    def stddev(self) -> float:
        return _core.poisson_stddev(self.lambda_)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.poisson_mgf_scalar(float(t), self.lambda_)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.poisson_cgf_scalar(float(t), self.lambda_)

    def sample(self) -> int:
        return _core.poisson_sample(self.lambda_)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, x: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(x=x)
        return _core.poisson_pmf_scalar(float(x), float(lambda_))

    @classmethod
    def _cdf_scalar(cls, x: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(x=x)
        return _core.poisson_cdf_scalar(float(x), float(lambda_))

    @classmethod
    def _mean(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_mean(float(lambda_))

    @classmethod
    def _variance(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_variance(float(lambda_))

    @classmethod
    def _stddev(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_stddev(float(lambda_))

    @classmethod
    def _mgf_scalar(cls, t: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(t=t)
        return _core.poisson_mgf_scalar(float(t), float(lambda_))

    @classmethod
    def _cgf_scalar(cls, t: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(t=t)
        return _core.poisson_cgf_scalar(float(t), float(lambda_))

    @classmethod
    def _sample(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_sample(float(lambda_))
