# python/distributions/gamma.py

try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Gamma:
    # Magic Methods
    __slots__ = ("alpha", "theta")

    def __init__(self, alpha=None, theta=None):
        self._validate_params(alpha=alpha, theta=theta)
        self.alpha = alpha
        self.theta = theta

    def __repr__(self):
        return f"Gamma(alpha={self.alpha}, theta={self.theta})"

    @staticmethod
    def _validate_params(alpha=None, theta=None):
        """Internal validation shared by all methods."""
        if alpha is not None:
            if alpha <= 0:
                raise ValueError("alpha must be positive")
        if theta is not None:
            if theta <= 0:
                raise ValueError("theta must be positive")

    # Instance Methods
    def pmf_scalar(self, x):
        return self._pdf_scalar(x, self.alpha, self.theta)

    def cdf_scalar(self, x):
        return self._cdf_scalar(x, self.alpha, self.theta)

    def mean(self):
        return self._mean(self.alpha, self.theta)

    def variance(self):
        return self._variance(self.alpha, self.theta)

    def stddev(self):
        return self._stddev(self.alpha, self.theta)

    def mgf_scalar(self, t):
        return self._mgf_scalar(t, self.alpha, self.theta)

    def cgf_scalar(self, t):
        return self._cgf_scalar(t, self.alpha, self.theta)

    def sample(self):
        return self._sample(self.alpha, self.theta)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_pdf_scalar(x, alpha, theta)

    @classmethod
    def _cdf_scalar(cls, x, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_cdf_scalar(x, alpha, theta)

    @classmethod
    def _mean(cls, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_mean(alpha, theta)

    @classmethod
    def _variance(cls, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_variance(alpha, theta)

    @classmethod
    def _stddev(cls, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_stddev(alpha, theta)

    @classmethod
    def _mgf_scalar(cls, t, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_mgf_scalar(t, alpha, theta)

    @classmethod
    def _cgf_scalar(cls, t, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_cgf_scalar(t, alpha, theta)

    @classmethod
    def _sample(cls, alpha, theta):
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_sample(alpha, theta)
