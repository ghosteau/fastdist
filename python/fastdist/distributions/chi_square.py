# python/distributions/discrete_uniform.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class ChiSquare:
    # Magic Methods
    __slots__ = "k"

    def __init__(self, k=None):
        ChiSquare._validate_params(k=k)
        self.k = k

    def __repr__(self):
        return f"ChiSquare(k={self.k})"

    @staticmethod
    def _validate_params(k=None):
        """Internal validation shared by all methods."""
        if k is not None:
            if k <= 0:
                raise ValueError("k must be positive")

    # Instance Methods
    def pmf(self, x):
        return self._pdf_scalar(x, self.k)

    def cdf(self, x):
        return self._cdf_scalar(x, self.k)

    def mean(self):
        return self._mean(self.k)

    def variance(self):
        return self._variance(self.k)

    def stddev(self):
        return self._stddev(self.k)

    def mgf_scalar(self, t):
        return self._mgf_scalar(t, self.k)

    def cgf_scalar(self, t):
        return self._cgf_scalar(t, self.k)

    def sample(self):
        return self._sample(self.k)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x, k):
        cls._validate_params(k)
        return _core.chi_square_pdf_scalar(x, k)

    @classmethod
    def _cdf_scalar(cls, x, k):
        cls._validate_params(k)
        return _core.chi_square_cdf_scalar(x, k)

    @classmethod
    def _mean(cls, k):
        cls._validate_params(k)
        return _core.chi_square_mean(k)

    @classmethod
    def _variance(cls, k):
        cls._validate_params(k)
        return _core.chi_square_variance(k)

    @classmethod
    def _stddev(cls, k):
        cls._validate_params(k)
        return _core.chi_square_stddev(k)

    @classmethod
    def _mgf_scalar(cls, t, k):
        cls._validate_params(k)
        return _core.chi_square_mgf_scalar(t, k)

    @classmethod
    def _cgf_scalar(cls, t, k):
        cls._validate_params(k)
        return _core.chi_square_cgf_scalar(t, k)

    @classmethod
    def _sample(cls, k):
        cls._validate_params(k)
        return _core.chi_square_sample(k)
