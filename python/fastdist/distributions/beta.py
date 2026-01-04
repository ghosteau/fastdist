# python/distributions/bernoulli.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Beta:
    # Magic Methods
    __slots__ = ("alpha", "beta")

    def __init__(self, alpha=None, beta=None):
        Beta._validate_params(alpha, beta)
        self.alpha = alpha
        self.beta = beta

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta})"

    @staticmethod
    def _validate_params(alpha=None, beta=None):
        """Internal validation shared by all methods."""
        if alpha is not None:
            if alpha <= 0:
                raise ValueError("alpha must be positive")
        if beta is not None:
            if beta <= 0:
                raise ValueError("beta must be positive")

    # Instance Methods
    def pdf_scalar(self, x):
        return Beta._pdf_scalar(x, self.alpha, self.beta)

    def cdf_scalar(self, x):
        return Beta._cdf_scalar(x, self.alpha, self.beta)

    def mean(self):
        return Beta._mean(self.alpha, self.beta)

    def variance(self):
        return Beta._variance(self.alpha, self.beta)

    def stddev(self):
        return Beta._stddev(self.alpha, self.beta)

    def sample(self):
        return Beta._sample(self.alpha, self.beta)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x, alpha, beta):
        cls._validate_params(alpha, beta)
        return _core.beta_pdf_scalar(x, alpha, beta)

    @classmethod
    def _cdf_scalar(cls, x, alpha, beta):
        cls._validate_params(alpha, beta)
        return _core.beta_cdf_scalar(x, alpha, beta)

    @classmethod
    def _mean(cls, alpha, beta):
        cls._validate_params(alpha, beta)
        return _core.beta_mean(alpha, beta)

    @classmethod
    def _variance(cls, alpha, beta):
        cls._validate_params(alpha, beta)
        return _core.beta_variance(alpha, beta)

    @classmethod
    def _stddev(cls, alpha, beta):
        cls._validate_params(alpha, beta)
        return _core.beta_stddev(alpha, beta)

    @classmethod
    def _sample(cls, alpha, beta):
        cls._validate_params(alpha, beta)
        return _core.beta_sample(alpha, beta)
