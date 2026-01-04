# python/distributions/normal.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import math


class Normal:
    # Magic Methods
    __slots__ = ("mu", "sigma")

    def __init__(self, mu: float, sigma: float):
        self._validate_params(mu=mu, sigma=sigma)
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    @staticmethod
    def _validate_params(mu: int | float = None, sigma: int | float = None):
        """Internal validation shared by all methods."""
        if mu is not None and not isinstance(mu, (int, float)):
            raise TypeError("mu must be a real number")
        if sigma is not None and not isinstance(sigma, (int, float)):
            raise TypeError("sigma must be a real number")
        if mu is not None and not math.isfinite(mu):
            raise ValueError("mu must be finite")
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None):
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ----------------
    # Instance Methods
    # ----------------
    def pdf_scalar(self, x) -> float:
        self._validate_inputs(x=x)
        return _core.normal_pdf_scalar(x, self.mu, self.sigma)

    def logpdf_scalar(self, x) -> float:
        self._validate_inputs(x=x)
        return _core.normal_logpdf_scalar(x, self.mu, self.sigma)

    def cdf_scalar(self, x) -> float:
        self._validate_inputs(x=x)
        return _core.normal_cdf_scalar(x, self.mu, self.sigma)

    def mean(self) -> float:
        return _core.normal_mean(self.mu)

    def variance(self) -> float:
        return _core.normal_variance(self.sigma)

    def stddev(self) -> float:
        return _core.normal_stddev(self.sigma)

    def mgf_scalar(self, t) -> float:
        self._validate_inputs(t=t)
        return _core.normal_mgf_scalar(t, self.mu, self.sigma)

    def cgf_scalar(self, t) -> float:
        self._validate_inputs(t=t)
        return _core.normal_cgf_scalar(t, self.mu, self.sigma)

    def sample(self) -> float:
        return _core.normal_sample(self.mu, self.sigma)

    def log_sample(self) -> float:
        return _core.normal_log_sample(self.mu, self.sigma)

    def z_score(self, x) -> float:
        self._validate_inputs(x=x)
        return _core.z_score(x, self.mu, self.sigma)

    # --------------
    # Static Methods
    # --------------
    @classmethod
    def _pdf_scalar(cls, x: int | float, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(x=x)
        return _core.normal_pdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def _logpdf_scalar(cls, x: int | float, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(x=x)
        return _core.normal_logpdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def _cdf_scalar(cls, x: int | float, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(x=x)
        return _core.normal_cdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def _mean(cls, mu: int | float) -> float:
        cls._validate_params(mu=mu)
        return _core.normal_mean(float(mu))

    @classmethod
    def _variance(cls, sigma: int | float) -> float:
        cls._validate_params(sigma=sigma)
        return _core.normal_variance(float(sigma))

    @classmethod
    def _stddev(cls, sigma: int | float) -> float:
        cls._validate_params(sigma=sigma)
        return _core.normal_stddev(float(sigma))

    @classmethod
    def _mgf_scalar(cls, t: int | float, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(t=t)
        return _core.normal_mgf_scalar(float(t), float(mu), float(sigma))

    @classmethod
    def _cgf_scalar(cls, t: int | float, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(t=t)
        return _core.normal_cgf_scalar(float(t), float(mu), float(sigma))

    @classmethod
    def _sample(cls, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_sample(float(mu), float(sigma))

    @classmethod
    def _log_sample(cls, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_log_sample(float(mu), float(sigma))

    @classmethod
    def _z_score(cls, x: int | float, mu: int | float, sigma: int | float) -> float:
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.z_score(float(x), float(mu), float(sigma))

    # ----------------------
    # Batch Instance Methods
    # ----------------------
    def pdf_cpu(self, x):
        return _core.pdf_cpu(x, self.mu, self.sigma)

    # --------------------
    # Batch Static Methods
    # --------------------
    @classmethod
    def _pdf_cpu(cls, x, mu, sigma):
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_pdf_cpu(x, mu, sigma)

    # ---------------------
    # CUDA Instance Methods
    # ---------------------
    def pdf_cuda(self, x):
        return Normal._pdf_cuda(x, self.mu, self.sigma)

    # -------------------
    # CUDA Static Methods
    # -------------------
    @classmethod
    def _pdf_cuda(cls, x, mu, sigma):
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_pdf_cuda(x, mu, sigma)
