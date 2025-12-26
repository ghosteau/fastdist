# python/distributions/poisson.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Poisson:
    # Magic Methods
    __slots__ = "lambda_"

    def __init__(self, lambda_=None):
        Poisson._validate_params(lambda_=lambda_)
        self.lambda_ = lambda_

    def __repr__(self):
        return f"Poisson(lambda_={self.lambda_})"

    @staticmethod
    def _validate_params(lambda_=None):
        """Internal validation shared by all methods."""
        if lambda_ is not None:
            if lambda_ <= 0:
                raise ValueError("lambda_ must be positive")

    # Instance Methods
    def pmf_scalar(self, k):
        return Poisson._pmf_scalar(k, self.lambda_)

    def cdf_scalar(self, k):
        return Poisson._cdf_scalar(k, self.lambda_)

    def mean(self):
        return Poisson._mean(self.lambda_)

    def variance(self):
        return Poisson._variance(self.lambda_)

    def stddev(self):
        return Poisson._stddev(self.lambda_)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_pmf_scalar(k, lambda_)

    @classmethod
    def _cdf_scalar(cls, k, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_cdf_scalar(k, lambda_)

    @classmethod
    def _mean(cls, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_mean(lambda_)

    @classmethod
    def _variance(cls, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_variance(lambda_)

    @classmethod
    def _stddev(cls, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_stddev(lambda_)