# python/distributions/exponential.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Exponential:
    # Magic Methods
    __slots__ = ("_lambda_",)

    def __init__(self, lambda_: int | float):
        self._validate_params(lambda_=lambda_)
        self._lambda_ = float(lambda_)

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value):
        self._validate_params(lambda_=value)
        self._lambda_ = float(value)

    def __repr__(self):
        return f"Exponential(lambda_={self.lambda_})"

    @staticmethod
    def _validate_params(lambda_: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(lambda_, (int, float)):
            raise TypeError("lambda_ must be a real number")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None, step_size=None) -> None:
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

    # Instance Methods
    def pdf_scalar(self, x: int | float) -> float:
        self._validate_inputs(x=x)
        return _core.exponential_pdf_scalar(float(x), self.lambda_)

    def cdf_scalar(self, x: int | float) -> float:
        self._validate_inputs(x=x)
        return _core.exponential_cdf_scalar(float(x), self.lambda_)

    def mean(self) -> float:
        return _core.exponential_mean(self.lambda_)

    def variance(self) -> float:
        return _core.exponential_variance(self.lambda_)

    def stddev(self) -> float:
        return _core.exponential_stddev(self.lambda_)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.exponential_mgf_scalar(float(t), self.lambda_)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.exponential_cgf_scalar(float(t), self.lambda_)

    def sample(self) -> float:
        return _core.exponential_sample(self.lambda_)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(x=x)
        return _core.exponential_pdf_scalar(float(x), float(lambda_))

    @classmethod
    def _cdf_scalar(cls, x: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(x=x)
        return _core.exponential_cdf_scalar(float(x), float(lambda_))

    @classmethod
    def _mean(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_mean(float(lambda_))

    @classmethod
    def _variance(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_variance(float(lambda_))

    @classmethod
    def _stddev(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_stddev(float(lambda_))

    @classmethod
    def _mgf_scalar(cls, t: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(t=t)
        return _core.exponential_mgf_scalar(float(t), float(lambda_))

    @classmethod
    def _cgf_scalar(cls, t: int | float, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(t=t)
        return _core.exponential_cgf_scalar(float(t), float(lambda_))

    @classmethod
    def _sample(cls, lambda_: int | float) -> float:
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_sample(float(lambda_))

    # ----------------------
    # Batch Instance Methods
    # ----------------------
    def pdf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.exponential_pdf_cpu(x, self._lambda_, step_size)

    def cdf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.exponential_cdf_cpu(x, self._lambda_, step_size)

    def mgf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.exponential_mgf_cpu(x, self._lambda_, step_size)

    def cgf_cpu(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(x)
        return _core.exponential_cgf_cpu(x, self._lambda_, step_size)

    # --------------------
    # Batch Static Methods
    # --------------------
    @classmethod
    def _pdf_cpu(cls, x: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(x)
        return _core.exponential_pdf_cpu(x, lambda_, step_size)

    @classmethod
    def _cdf_cpu(cls, x: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=x)
        return _core.exponential_cdf_cpu(x, lambda_, step_size)

    @classmethod
    def _mgf_cpu(cls, t: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=t)
        return _core.exponential_mgf_cpu(t, lambda_, step_size)

    @classmethod
    def _cgf_cpu(cls, t: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=t)
        return _core.exponential_cgf_cpu(t, lambda_, step_size)

    # ---------------------
    # CUDA Instance Methods
    # ---------------------
    def pdf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.exponential_pdf_cuda(x, self._lambda_, step_size)

    def cdf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.exponential_cdf_cuda(x, self._lambda_, step_size)

    def mgf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.exponential_mgf_cuda(x, self._lambda_, step_size)

    def cgf_cuda(self, x: ArrayLike, step_size: float = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=x)
        return _core.exponential_cgf_cuda(x, self._lambda_, step_size)

    # -------------------
    # CUDA Static Methods
    # -------------------
    @classmethod
    def _pdf_cuda(cls, x: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=x)
        return _core.exponential_pdf_cuda(x, lambda_, step_size)

    @classmethod
    def _cdf_cuda(cls, x: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=x)
        return _core.exponential_cdf_cuda(x, lambda_, step_size)

    @classmethod
    def _mgf_cuda(cls, x: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=x)
        return _core.exponential_mgf_cuda(x, lambda_, step_size)

    @classmethod
    def _cgf_cuda(cls, x: ArrayLike, lambda_: float, step_size: float = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        cls._validate_array(arr=x)
        return _core.exponential_cgf_cuda(x, lambda_, step_size)
