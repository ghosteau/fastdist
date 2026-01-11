# python/distributions/bernoulli.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Bernoulli:
    # Magic Methods
    __slots__ = ("_p",)

    def __init__(self, p: int | float):
        self._validate_params(p=p)
        self._p = float(p)

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._validate_params(p=value)
        self._p = float(value)

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    @staticmethod
    def _validate_params(p: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if p < 0 or p > 1:
            raise ValueError("p must be in the interval [0, 1]")

    @staticmethod
    def _validate_inputs(k=None, t=None, step_size=None) -> None:
        if k is not None and not isinstance(k, int):
            raise TypeError("k must be an integer")
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
    def pmf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.bernoulli_pmf_scalar(k, self.p)

    def cdf_scalar(self, k: int) -> float:
        self._validate_inputs(k=k)
        return _core.bernoulli_cdf_scalar(k, self.p)

    def mean(self) -> float:
        return _core.bernoulli_mean(self.p)

    def variance(self) -> float:
        return _core.bernoulli_variance(self.p)

    def stddev(self) -> float:
        return _core.bernoulli_stddev(self.p)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.bernoulli_mgf(float(t), self.p)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.bernoulli_cgf(float(t), self.p)

    def sample(self) -> int:
        return _core.bernoulli_sample(self.p)

    # Static Methods
    @classmethod
    def _pmf_scalar(cls, k: int, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.bernoulli_pmf_scalar(k, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(k=k)
        return _core.bernoulli_cdf_scalar(k, float(p))

    @classmethod
    def _mean(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.bernoulli_mean(float(p))

    @classmethod
    def _variance(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.bernoulli_variance(float(p))

    @classmethod
    def _stddev(cls, p: int | float) -> float:
        cls._validate_params(p=p)
        return _core.bernoulli_stddev(float(p))

    @classmethod
    def _mgf_scalar(cls, t: int | float, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(t=t)
        return _core.bernoulli_mgf(float(t), float(p))

    @classmethod
    def _cgf_scalar(cls, t: int | float, p: int | float) -> float:
        cls._validate_params(p=p)
        cls._validate_inputs(t=t)
        return _core.bernoulli_cgf(float(t), float(p))

    @classmethod
    def _sample(cls, p: int | float) -> int:
        cls._validate_params(p=p)
        return _core.bernoulli_sample(float(p))

    # ----------------------
    # Batch Instance Methods
    # ----------------------
    def pmf_cpu(self, k: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=k)
        return _core.bernoulli_pmf_cpu(k, self.p, step_size)

    def cdf_cpu(self, k: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=k)
        return _core.bernoulli_cdf_cpu(k, self.p, step_size)

    def mgf_cpu(self, t: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=t)
        return _core.bernoulli_mgf_cpu(t, self.p, step_size)

    def cgf_cpu(self, t: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=t)
        return _core.bernoulli_cgf_cpu(t, self.p, step_size)

    # --------------------
    # Batch Static Methods
    # --------------------
    @classmethod
    def _pmf_cpu(cls, k: ArrayLike, p: float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=k)
        return _core.bernoulli_pmf_cpu(k, p, step_size)

    @classmethod
    def _cdf_cpu(cls, k: ArrayLike, p: float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=k)
        return _core.bernoulli_cdf_cpu(k, p, step_size)

    @classmethod
    def _mgf_cpu(cls, t: ArrayLike, p: int | float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=t)
        return _core.bernoulli_mgf_cpu(t, p, step_size)

    @classmethod
    def _cgf_cpu(cls, t: ArrayLike, p: int | float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=t)
        return _core.bernoulli_cgf_cpu(t, p, step_size)

    # ---------------------
    # CUDA Instance Methods
    # ---------------------
    def pmf_cuda(self, k: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=k)
        return _core.bernoulli_pmf_cuda(k, self.p, step_size)

    def cdf_cuda(self, k: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=k)
        return _core.bernoulli_cdf_cuda(k, self.p, step_size)

    def mgf_cuda(self, t: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=t)
        return _core.bernoulli_mgf_cuda(t, self.p, step_size)

    def cgf_cuda(self, t: ArrayLike, step_size: int = 0) -> NDArray[np.float64]:
        self._validate_inputs(step_size=step_size)
        self._validate_array(arr=t)
        return _core.bernoulli_cgf_cuda(t, self.p, step_size)

    # -------------------
    # CUDA Static Methods
    # -------------------
    @classmethod
    def _pmf_cuda(cls, k: ArrayLike, p: int | float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=k)
        return _core.bernoulli_pmf_cuda(k, p, step_size)

    @classmethod
    def _cdf_cuda(cls, k: ArrayLike, p: int | float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=k)
        return _core.bernoulli_cdf_cuda(k, p, step_size)

    @classmethod
    def _mgf_cuda(cls, t: ArrayLike, p: int | float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=t)
        return _core.bernoulli_mgf_cuda(t, p, step_size)

    @classmethod
    def _cgf_cuda(cls, t: ArrayLike, p: int | float, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(step_size=step_size)
        cls._validate_params(p=p)
        cls._validate_array(arr=t)
        return _core.bernoulli_cgf_cuda(t, p, step_size)
