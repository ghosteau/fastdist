# python/distributions/exponential.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Exponential:
    # Magic Methods
    __slots__ = "lambda_"

    def __init__(self, lambda_=None):
        Exponential._validate_params(lambda_=lambda_)
        self.lambda_ = lambda_

    def __repr__(self):
        return f"Exponential(lambda_={self.lambda_})"

    @staticmethod
    def _validate_params(lambda_=None):
        """Internal validation shared by all methods."""
        if lambda_ is not None:
            if lambda_ <= 0:
                raise ValueError("lambda_ must be positive")

    # Instance Methods
    def pdf_scalar(self, x):
        return Exponential._pdf_scalar(x, self.lambda_)

    def logpdf_scalar(self, x):
        return Exponential._logpdf_scalar(x, self.lambda_)

    def cdf_scalar(self, x):
        return Exponential._cdf_scalar(x, self.lambda_)

    def mean(self):
        return Exponential._mean(self.lambda_)

    def variance(self):
        return Exponential._variance(self.lambda_)

    def stddev(self):
        return Exponential._stddev(self.lambda_)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_pdf_scalar(x, lambda_)

    @classmethod
    def _logpdf_scalar(cls, x, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_logpdf_scalar(x, lambda_)

    @classmethod
    def _cdf_scalar(cls, x, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_cdf_scalar(x, lambda_)

    @classmethod
    def _mean(cls, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_mean(lambda_)

    @classmethod
    def _variance(cls, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_variance(lambda_)

    @classmethod
    def _stddev(cls, lambda_):
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_stddev(lambda_)
