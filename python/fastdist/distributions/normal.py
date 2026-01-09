# python/distributions/normal.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Normal:
    # Magic Methods
    __slots__ = ("_mu", "_sigma")

    def __init__(self, mu: float, sigma: float):
        self._validate_params(mu=mu, sigma=sigma)
        self._mu = float(mu)
        self._sigma = float(sigma)

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @mu.setter
    def mu(self, value):
        self._validate_params(mu=value)
        self._mu = float(value)

    @sigma.setter
    def sigma(self, value):
        self._validate_params(sigma=value)
        self._sigma = float(value)

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    @staticmethod
    def _validate_params(mu: int | float = None, sigma: int | float = None):
        """Internal validation shared by all methods."""
        if mu is not None:
            if not isinstance(mu, (int, float)):
                raise TypeError("mu must be a real number")
            if not math.isfinite(mu):
                raise ValueError("mu must be finite")
        if sigma is not None:
            if not isinstance(sigma, (int, float)):
                raise TypeError("sigma must be a real number")
            if not math.isfinite(sigma):
                raise ValueError("sigma must be finite")
            if sigma <= 0:
                raise ValueError("sigma must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None, step_size=None):
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")
        if step_size is not None and not isinstance(step_size, (int, float)):
            raise TypeError("step_size must be a real number")

    @staticmethod
    def _validate_array(arr):
        # Convert into Numpy array
        temp_arr = np.asarray(arr, dtype=np.float64)

        if not np.issubdtype(temp_arr.dtype, np.floating):
            raise TypeError("Array must be numeric")
        if temp_arr.ndim != 1:
            raise ValueError("Array must be 1-dimensional")

    # ----------------
    # Instance Methods
    # ----------------
    def pdf_scalar(self, x: float) -> float:
        self._validate_inputs(x=x)
        return _core.normal_pdf_scalar(x, self.mu, self.sigma)

    def logpdf_scalar(self, x: float) -> float:
        self._validate_inputs(x=x)
        return _core.normal_logpdf_scalar(x, self.mu, self.sigma)

    def cdf_scalar(self, x: float) -> float:
        self._validate_inputs(x=x)
        return _core.normal_cdf_scalar(x, self.mu, self.sigma)

    def mean(self) -> float:
        return _core.normal_mean(self.mu)

    def variance(self) -> float:
        return _core.normal_variance(self.sigma)

    def stddev(self) -> float:
        return _core.normal_stddev(self.sigma)

    def mgf_scalar(self, t: float) -> float:
        self._validate_inputs(t=t)
        return _core.normal_mgf_scalar(t, self.mu, self.sigma)

    def cgf_scalar(self, t: float) -> float:
        self._validate_inputs(t=t)
        return _core.normal_cgf_scalar(t, self.mu, self.sigma)

    def sample(self) -> float:
        return _core.normal_sample(self.mu, self.sigma)

    def log_sample(self) -> float:
        return _core.normal_log_sample(self.mu, self.sigma)

    def z_score(self, x: float) -> float:
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
    def pdf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.normal_pdf_cpu(x, self.mu, self.sigma, step_size)

    def logpdf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.normal_logpdf_cpu(x, self.mu, self.sigma, step_size)

    def cdf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.normal_cdf_cpu(x, self.mu, self.sigma, step_size)

    def mgf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.normal_mgf_cpu(x, self.mu, self.sigma, step_size)

    def cgf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.normal_cgf_cpu(x, self.mu, self.sigma, step_size)

    # --------------------
    # Batch Static Methods
    # --------------------
    @classmethod
    def _pdf_cpu(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(x)
        return _core.normal_pdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def _logpdf_cpu(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_logpdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def _cdf_cpu(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_cdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def _mgf_cpu(cls, t: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=t)
        return _core.normal_mgf_cpu(t, mu, sigma, step_size)

    @classmethod
    def _cgf_cpu(cls, t: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=t)
        return _core.normal_cgf_cpu(t, mu, sigma, step_size)

    # ---------------------
    # CUDA Instance Methods
    # ---------------------
    def pdf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.normal_pdf_cuda(x, self.mu, self.sigma, step_size)

    def logpdf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.normal_logpdf_cuda(x, self.mu, self.sigma, step_size)

    def cdf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.normal_cdf_cuda(x, self.mu, self.sigma, step_size)

    def mgf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.normal_mgf_cuda(x, self.mu, self.sigma, step_size)

    def cgf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.normal_cgf_cuda(x, self.mu, self.sigma, step_size)

    # -------------------
    # CUDA Static Methods
    # -------------------
    @classmethod
    def _pdf_cuda(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_pdf_cuda(x, mu, sigma, step_size)

    @classmethod
    def _logpdf_cuda(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_logpdf_cuda(x, mu, sigma, step_size)

    @classmethod
    def _cdf_cuda(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_cdf_cuda(x, mu, sigma, step_size)

    @classmethod
    def _mgf_cuda(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_mgf_cuda(x, mu, sigma, step_size)

    @classmethod
    def _cgf_cuda(cls, x: ArrayLike, mu: float, sigma: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_array(arr=x)
        return _core.normal_cgf_cuda(x, mu, sigma, step_size)
