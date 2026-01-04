# python/distributions/discrete_uniform.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class DiscreteUniform:
    # Magic Methods
    __slots__ = ("a", "b")

    def __init__(self, a=None, b=None):
        DiscreteUniform._validate_params(a=a, b=b)
        self.a = a
        self.b = b

    def __repr__(self):
        return f"DiscreteUniform(a={self.a}, b={self.b})"

    @staticmethod
    def _validate_params(a=None, b=None):
        """Internal validation shared by all methods."""
        if a is not None and b is not None:
            if a >= b:
                raise ValueError("a must be less than b")

    # Instance Methods
    def pmf(self, x):
        return self._pmf_scalar(x, self.a, self.b)

    def cdf(self, x):
        return self._cdf_scalar(x, self.a, self.b)

    def mean(self):
        return self._mean(self.a, self.b)

    def variance(self):
        return self._variance(self.a, self.b)

    def stddev(self):
        return self._stddev(self.a, self.b)

    def mgf(self, t):
        return self._mgf_scalar(t, self.a, self.b)

    def cgf(self, t):
        return self._cgf_scalar(t, self.a, self.b)

    def sample(self):
        return self._sample(self.a, self.b)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, x, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_pmf_scalar(x, a, b)

    @classmethod
    def _cdf_scalar(cls, x, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_cdf_scalar(x, a, b)

    @classmethod
    def _mean(cls, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_mean(a, b)

    @classmethod
    def _variance(cls, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_variance(a, b)

    @classmethod
    def _stddev(cls, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_stddev(a, b)

    @classmethod
    def _mgf_scalar(cls, t, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_mgf_scalar(t, a, b)

    @classmethod
    def _cgf_scalar(cls, t, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_cgf_scalar(t, a, b)

    @classmethod
    def _sample(cls, a, b):
        cls._validate_params(a, b)
        return _core.discrete_uniform_sample(a, b)
