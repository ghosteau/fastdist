# python/distributions/bernoulli.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Bernoulli:
    # Magic Methods
    __slots__ = ("p")

    def __init__(self, p=None):
        Bernoulli._validate_params(p)
        self.p = p

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    @staticmethod
    def _validate_params(p=None):
        """Internal validation shared by all methods."""
        if p is not None:
            if p < 0 or p > 1:
                raise ValueError("p must be in the interval [0, 1]")

    # Instance Methods
    def pmf_scalar(self, k):
        return Bernoulli._pmf_scalar(k, self.p)

    def cdf_scalar(self, k):
        return Bernoulli._cdf_scalar(k, self.p)

    def mean(self):
        return Bernoulli._mean(self.p)

    def variance(self):
        return Bernoulli._variance(self.p)

    def stddev(self):
        return Bernoulli._stddev(self.p)

    def mgf_scalar(self, t):
        return Bernoulli._mgf_scalar(t, self.p)

    def cgf_scalar(self, t):
        return Bernoulli._cgf_scalar(t, self.p)

    def sample(self):
        return Bernoulli._sample(self.p)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k, p):
        cls._validate_params(p)
        return _core.bernoulli_pmf_scalar(k, p)

    @classmethod
    def _cdf_scalar(cls, k, p):
        cls._validate_params(p)
        return _core.bernoulli_cdf_scalar(k, p)

    @classmethod
    def _mean(cls, p):
        cls._validate_params(p)
        return _core.bernoulli_mean(p)

    @classmethod
    def _variance(cls, p):
        cls._validate_params(p)
        return _core.bernoulli_variance(p)

    @classmethod
    def _stddev(cls, p):
        cls._validate_params(p)
        return _core.bernoulli_stddev(p)

    @classmethod
    def _mgf_scalar(cls, t, p):
        cls._validate_params(p)
        return _core.bernoulli_mgf(t, p)

    @classmethod
    def _cgf_scalar(cls, t, p):
        cls._validate_params(p)
        return _core.bernoulli_cgf(t, p)

    @classmethod
    def _sample(cls, p):
        cls._validate_params(p)
        return _core.bernoulli_sample(p)
