# python/distributions/normal.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Normal:
    # Magic Methods
    __slots__ = ("mu", "sigma")

    def __init__(self, mu=None, sigma=None):
        Normal._validate_params(sigma=sigma)
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    @staticmethod
    def _validate_params(sigma=None):
        """Internal validation shared by all methods."""
        if sigma is not None:
            if sigma <= 0:
                raise ValueError("sigma must be positive")

    # Instance Methods
    def pdf_scalar(self, x):
        return Normal._pdf_scalar(x, self.mu, self.sigma)

    def logpdf_scalar(self, x):
        return Normal._logpdf_scalar(x, self.mu, self.sigma)

    def cdf_scalar(self, x):
        return Normal._cdf_scalar(x, self.mu, self.sigma)

    def mean(self):
        return Normal._mean(self.mu)

    def variance(self):
        return Normal._variance(self.sigma)

    def stddev(self):
        return Normal._stddev(self.sigma)

    def z_score(self, x):
        return Normal._z_score(x, self.mu, self.sigma)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x, mu, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_pdf_scalar(x, mu, sigma)

    @classmethod
    def _logpdf_scalar(cls, x, mu, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_logpdf_scalar(x, mu, sigma)

    @classmethod
    def _cdf_scalar(cls, x, mu, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_cdf_scalar(x, mu, sigma)

    @classmethod
    def _mean(cls, mu):
        return _core.normal_mean(mu)

    @classmethod
    def _variance(cls, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_variance(sigma)

    @classmethod
    def _stddev(cls, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_stddev(sigma)

    @classmethod
    def _z_score(cls, x, mu, sigma):
        cls._validate_params(sigma=sigma)
        return _core.z_score(x, mu, sigma)

    # Batch Instance Methods
    def pdf_cpu(self, x):
        return Normal._pdf_cpu(x, self.mu, self.sigma)

    # Batch Static Methods
    @classmethod
    def _pdf_cpu(cls, x, mu, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_pdf_cpu(x, mu, sigma)

    # CUDA Instance Methods
    def pdf_cuda(self, x):
        return Normal._pdf_cuda(x, self.mu, self.sigma)

    # CUDA Static Methods
    @classmethod
    def _pdf_cuda(cls, x, mu, sigma):
        cls._validate_params(sigma=sigma)
        return _core.normal_pdf_cuda(x, mu, sigma)
