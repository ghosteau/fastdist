# python/distributions/poisson.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class NegativeBinomial:
    # Magic Methods
    __slots__ = ("r", "p")

    def __init__(self, r=None, p=None):
        self._validate_params(r=r, p=p)
        self.r = r
        self.p = p

    def __repr__(self):
        return f"NegativeBinomial(r={self.r}, p={self.p})"

    @staticmethod
    def _validate_params(r=None, p=None):
        """Internal validation shared by all methods."""
        if r is not None:
            if r <= 0:
                raise ValueError("r must be positive")
        if p is not None:
            if p < 0 or p > 1:
                raise ValueError("p must be in [0, 1]")

    # Instance Methods
    def pmf_scalar(self, k):
        return self._pmf_scalar(k, self.r, self.p)

    def cdf_scalar(self, k):
        return self._cdf_scalar(k, self.r, self.p)

    def mean(self):
        return self._mean(self.r, self.p)

    def variance(self):
        return self._variance(self.r, self.p)

    def stddev(self):
        return self._stddev(self.r, self.p)

    def mgf_scalar(self, t):
        return self._mgf_scalar(t, self.r, self.p)

    def cgf_scalar(self, t):
        return self._cgf_scalar(t, self.r, self.p)

    def sample(self):
        return self._sample(self.r, self.p)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_pmf_scalar(k, r, p)

    @classmethod
    def _cdf_scalar(cls, k, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_cdf_scalar(k, r, p)

    @classmethod
    def _mean(cls, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_mean(r, p)

    @classmethod
    def _variance(cls, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_variance(r, p)

    @classmethod
    def _stddev(cls, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_stddev(r, p)

    @classmethod
    def _mgf_scalar(cls, t, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_mgf_scalar(t, r, p)

    @classmethod
    def _cgf_scalar(cls, t, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_cgf_scalar(t, r, p)

    @classmethod
    def _sample(cls, r, p):
        cls._validate_params(r=r, p=p)
        return _core.negative_binomial_sample(r, p)
