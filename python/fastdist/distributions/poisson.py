# python/distributions/poisson.py
try:
    from fastdist import _fastdist as _core
    from fastdist import config
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

from numbers import Real
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'poisson_pmf_cuda')


class Poisson:
    # Magic Methods
    __slots__ = ("_lambda_",)

    def __init__(self, lambda_: Real):
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
        return f"Poisson(lambda_={self.lambda_})"

    @staticmethod
    def _validate_params(lambda_: Real) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(lambda_, Real):
            raise TypeError("lambda_ must be a real number")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

    @staticmethod
    def _validate_inputs(_input: Union[Real, Sequence[Real]], input_name: str, step_size: Union[Real, None] = None) \
            -> Union[Real, np.ndarray]:
        if _input is None:
            raise TypeError(f"{input_name} cannot be None")

        if isinstance(_input, Real):
            validated = float(_input)
        else:
            validated = Poisson._validate_array(arr=_input, input_name=input_name)
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("step_size must be an integer")

        return validated

    @staticmethod
    def _validate_array(arr: Sequence[Real], input_name: str) -> np.ndarray:
        """
        Convert a sequence to a validated 1D NumPy array.

        Parameters
        ----------
        arr : sequence of Real
            Input array-like to validate.
        input_name : str
            Name of the input variable.

        Returns
        -------
        np.ndarray
            A 1D NumPy array of type float64.

        Raises
        ------
        TypeError
            If the array cannot be converted to floats.
        ValueError
            If the array is not 1-dimensional.

        Notes
        -----
        Used internally to standardize array-like inputs for CPU and CUDA operations.
        """

        try:
            arr = np.atleast_1d(arr).astype(np.float64)
        except (ValueError, TypeError):
            raise TypeError(f"{input_name} must be numeric (Real, or array-like of numbers)")

        if arr.ndim != 1:
            raise ValueError(f"{input_name} must be 1-dimensional")

        return arr

    @classmethod
    def is_cuda_available(cls) -> bool:
        """
        Check if CUDA acceleration is available.

        Returns
        -------
        bool
            True if CUDA functions are available, False otherwise.

        Notes
        -----
        Used internally to optimize performance for large arrays.
        """
        return _CUDA_AVAILABLE

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pmf(self, x: Union[Real, Sequence[Real]],
            step_size: int = 0) -> Union[float, np.ndarray]:
        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)

        if isinstance(validated_input, Real):
            return _core.poisson_pmf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_pmf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_pmf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_pmf_cpu(validated_input, self.lambda_, step_size)

    def cdf(self, x: Union[Real, Sequence[Real]],
            step_size: int = 0) -> Union[float, np.ndarray]:
        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)

        if isinstance(validated_input, Real):
            return _core.poisson_cdf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_cdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cdf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_cdf_cpu(validated_input, self.lambda_, step_size)

    def mean(self, lambda_: Union[Real, None] = None) -> Real:
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_mean(lambda_)

    def variance(self, lambda_: Union[Real, None] = None) -> Real:
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_variance(lambda_)

    def stddev(self, lambda_: Union[Real, None] = None) -> Real:
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_stddev(lambda_)

    def mgf(self, t: Union[Real, Sequence[Real]],
            step_size: int = 0) -> Union[float, np.ndarray]:
        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.poisson_mgf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_mgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_mgf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_mgf_cpu(validated_input, self.lambda_, step_size)

    def cgf(self, t: Union[Real, Sequence[Real]],
            step_size: int = 0) -> Union[float, np.ndarray]:
        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)

        if isinstance(validated_input, Real):
            return _core.poisson_cgf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_cgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cgf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_cgf_cpu(validated_input, self.lambda_, step_size)

    def sample(self, lambda_: Union[Real, None] = None) -> int:
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_sample(lambda_)

    # ------------------------------------------------------------------------------------------------------------------
    # Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, x: Real, lambda_: Real) -> Real:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.poisson_pmf_scalar(float(x), float(lambda_))

    @classmethod
    def _cdf_scalar(cls, x: Real, lambda_: Real) -> Real:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.poisson_cdf_scalar(float(x), float(lambda_))

    @classmethod
    def _mgf_scalar(cls, t: Real, lambda_: Real) -> Real:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.poisson_mgf_scalar(float(t), float(lambda_))

    @classmethod
    def _cgf_scalar(cls, t: Real, lambda_: Real) -> Real:
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.poisson_cgf_scalar(float(t), float(lambda_))

    # ------------------------------------------------------------------------------------------------------------------
    # Batch Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_cpu(cls, x: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_pmf_cpu(x, lambda_, step_size)

    @classmethod
    def _cdf_cpu(cls, x: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_cdf_cpu(x, lambda_, step_size)

    @classmethod
    def _mgf_cpu(cls, t: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_mgf_cpu(t, lambda_, step_size)

    @classmethod
    def _cgf_cpu(cls, t: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_cgf_cpu(t, lambda_, step_size)

    # ------------------------------------------------------------------------------------------------------------------
    # CUDA Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def _pmf_cuda(cls, x: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_pmf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _cdf_cuda(cls, x: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cdf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _mgf_cuda(cls, t: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_mgf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _cgf_cuda(cls, t: Sequence[Real], lambda_: Real, step_size: int = 0) -> NDArray[np.float64]:
            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cgf_cuda(validated_input, lambda_, step_size)
    else:
        @classmethod
        def _pmf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _cdf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _mgf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _cgf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )
