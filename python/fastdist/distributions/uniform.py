# python/distributions/uniform.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Uniform:
    # Magic Methods
    __slots__ = ("a", "b")

    def __init__(self, a=None, b=None):
        Uniform._validate_params(a=a, b=b)
        self.a = a
        self.b = b

    def __repr__(self):
        return f"Uniform(a={self.a}, b={self.b})"

    @staticmethod
    def _validate_params(a=None, b=None):
        """Internal validation shared by all methods."""
        if a is not None and b is not None:
            if a >= b:
                raise ValueError("a must be less than b")

    # Instance Methods
    def pdf_scalar(self, x):
        return Uniform._pdf_scalar(x, self.a, self.b)

    def cdf_scalar(self, x):
        return Uniform._cdf_scalar(x, self.a, self.b)

    def mean(self):
        return Uniform._mean(self.a, self.b)

    def variance(self):
        return Uniform._variance(self.a, self.b)

    def stddev(self):
        return Uniform._stddev(self.a, self.b)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x, a, b):
        cls._validate_params(a=a, b=b)
        return _core.uniform_pdf_scalar(x, a, b)

    @classmethod
    def _cdf_scalar(cls, x, a, b):
        cls._validate_params(a=a, b=b)
        return _core.uniform_cdf_scalar(x, a, b)

    @classmethod
    def _mean(cls, a, b):
        cls._validate_params(a=a, b=b)
        return _core.uniform_mean(a, b)

    @classmethod
    def _variance(cls, a, b):
        cls._validate_params(a=a, b=b)
        return _core.uniform_variance(a, b)

    @classmethod
    def _stddev(cls, a, b):
        cls._validate_params(a=a, b=b)
        return _core.uniform_stddev(a, b)
