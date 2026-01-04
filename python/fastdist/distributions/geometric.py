# python/distributions/geometric.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Geometric:
    # Magic Methods
    __slots__ = "p"

    def __init__(self, p=None):
        Geometric._validate_params(p=p)
        self.p = p

    def __repr__(self):
        return f"Geometric(p={self.p})"

    @staticmethod
    def _validate_params(p=None):
        """Internal validation shared by all methods."""
        if p is not None:
            if not (0 < p <= 1):
                raise ValueError("p must be in the interval (0, 1]")

    # Instance Methods
    def pmf_scalar(self, k):
        return Geometric._pmf_scalar(k, self.p)

    def cdf_scalar(self, k):
        return Geometric._cdf_scalar(k, self.p)

    def mean(self):
        return Geometric._mean(self.p)

    def variance(self):
        return Geometric._variance(self.p)

    def stddev(self):
        return Geometric._stddev(self.p)

    def mgf_scalar(self, t):
        return Geometric._mgf_scalar(t, self.p)

    def cgf_scalar(self, t):
        return Geometric._cgf_scalar(t, self.p)

    def sample(self):
        return Geometric._sample(self.p)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k, p):
        cls._validate_params(p=p)
        return _core.geometric_pmf_scalar(k, p)

    @classmethod
    def _cdf_scalar(cls, k, p):
        cls._validate_params(p=p)
        return _core.geometric_cdf_scalar(k, p)

    @classmethod
    def _mean(cls, p):
        cls._validate_params(p=p)
        return _core.geometric_mean(p)

    @classmethod
    def _variance(cls, p):
        cls._validate_params(p=p)
        return _core.geometric_variance(p)

    @classmethod
    def _stddev(cls, p):
        cls._validate_params(p=p)
        return _core.geometric_stddev(p)

    @classmethod
    def _mgf_scalar(cls, t, p):
        cls._validate_params(p=p)
        return _core.geometric_mgf_scalar(t, p)

    @classmethod
    def _cgf_scalar(cls, t, p):
        cls._validate_params(p=p)
        return _core.geometric_cgf_scalar(t, p)

    @classmethod
    def _sample(cls, p):
        cls._validate_params(p=p)
        return _core.geometric_sample(p)
