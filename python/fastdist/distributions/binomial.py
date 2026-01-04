# python/distributions/binomial.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Binomial:
    # Magic Methods
    __slots__ = ("n", "p")

    def __init__(self, n=None, p=None):
        Binomial._validate_params(n=n, p=p)
        self.n = n
        self.p = p

    def __repr__(self):
        return f"Normal(mu={self.n}, sigma={self.p})"

    @staticmethod
    def _validate_params(n=None, p=None):
        """Internal validation shared by all methods."""
        if n is not None:
            if not isinstance(n, int):
                raise ValueError("Did not receive an integer, n must be a non-negative integer")
            if n < 0:
                raise ValueError("n must be a non-negative integer")
        if p is not None:
            if p < 0 or p > 1:
                raise ValueError("p must be in the interval [0, 1]")

    # Instance Methods
    def logpmf_scalar(self, x):
        return Binomial._logpmf_scalar(x, self.n, self.p)

    def pmf_scalar(self, x):
        return Binomial._pmf_scalar(x, self.n, self.p)

    def cdf_scalar(self, x):
        return Binomial._cdf_scalar(x, self.n, self.p)

    def mean(self):
        return Binomial._mean(self.n, self.p)

    def variance(self):
        return Binomial._variance(self.n, self.p)

    def stddev(self):
        return Binomial._stddev(self.n, self.p)

    def mgf_scalar(self, t):
        return Binomial._mgf_scalar(t, self.n, self.p)

    def cgf_scalar(self):
        return Binomial._cgf_scalar(t, self.n, self.p)

    def sample(self):
        return Binomial._sample(self.n, self.p)

    # Static Methods
    @classmethod
    def _logpmf_scalar(cls, x, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_logpmf_scalar(x, n, p)

    @classmethod
    def _pmf_scalar(cls, x, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_pmf_scalar(x, n, p)

    @classmethod
    def _cdf_scalar(cls, x, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_cdf_scalar(x, n, p)

    @classmethod
    def _mean(cls, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_mean(n, p)

    @classmethod
    def _variance(cls, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_variance(n, p)

    @classmethod
    def _stddev(cls, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_stddev(n, p)

    @classmethod
    def _mgf_scalar(cls, t, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_mgf_scalar(t, n, p)

    @classmethod
    def _cgf_scalar(cls, t, n, p):
        cls._validate_params(n=n, p=p)
        return _core.binomial_cgf_scalar(t, n, p)

    @classmethod
    def _sample(cls, n, p):
        cls._validate_params(n, p)
        return _core.binomial_sample(n, p)
