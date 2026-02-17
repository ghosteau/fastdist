# python/distributions/utils.py
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
_CUDA_AVAILABLE = hasattr(_core, 'sigmoid_cuda')


class Utils:
    @staticmethod
    def _validate_input(_input: Union[Real, Sequence[Real]], input_name: str, input_type: type, dims: int = None) -> \
            Union[Real, np.ndarray]:
        if _input is None:
            raise TypeError(f"{input_name} must not be None")
        if isinstance(_input, Sequence) and not isinstance(_input, (str, bytes)):
            dims = 1 if dims is None else dims
        if input_name in (None, ""):
            raise ValueError("input_name must be a non-empty string")

        if input_type is int:
            if not isinstance(_input, int):
                raise TypeError(f"{input_name} must be an integer")
            return _input

        if input_type in (Real, Sequence):
            if isinstance(_input, Real):
                validated = _input
            elif isinstance(_input, (Sequence, np.ndarray)) and not isinstance(_input, (str, bytes)):
                dims = 1 if dims is None else dims
                if dims not in (1, 2):
                    raise ValueError("dims must be 1 or 2")
                validated = Utils._validate_array(arr=_input, arr_name=input_name, dims=dims)
            else:
                raise TypeError(f"{input_name} must be Real or a sequence of Real numbers")
        else:
            raise TypeError("input_type must be Real")

        return validated

    @staticmethod
    def _validate_array(arr: Sequence[Real], arr_name: str, dims: int) -> NDArray[np.float64]:
        if arr_name in (None, ""):
            raise ValueError("arr_name must be a non-empty string")
        if dims not in (1, 2):
            raise ValueError("dims must be 1 or 2")

        try:
            arr = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(f"{arr} must be numeric")

        if arr.ndim != dims:
            raise ValueError(f"{arr_name} must be {dims}D")

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

    @classmethod
    def chebyshev_bound(cls, variance: Real, k: Real) -> float:
        cls._validate_input(_input=variance, input_name="variance", input_type=Real)
        cls._validate_input(_input=k, input_name="k", input_type=Real)
        return _core.chebyshev_bound(float(variance), float(k))

    @classmethod
    def bayes_rule(cls, p_B_given_A: Real, p_A: Real, p_B: Real) -> float:
        cls._validate_input(_input=p_B_given_A, input_name="p_B_given_A", input_type=Real)
        cls._validate_input(_input=p_A, input_name="p_A", input_type=Real)
        cls._validate_input(_input=p_B, input_name="p_B", input_type=Real)
        return _core.bayes_rule(float(p_B_given_A), float(p_A), float(p_B))

@classmethod
def law_of_total_probability(cls, p_A: Union[Real, Sequence[Real]],
                             p_B_given_A: Union[Real, Sequence[Real]]) -> float:
    # Use _validate_input to allow Real or sequence
    p_A_valid = cls._validate_input(_input=p_A, input_name="p_A", input_type=Sequence)
    p_B_given_A_valid = cls._validate_input(_input=p_B_given_A, input_name="p_B_given_A", input_type=Sequence)

    # Convert to list if numpy array
    if isinstance(p_A_valid, np.ndarray):
        p_A_valid = p_A_valid.tolist()
    if isinstance(p_B_given_A_valid, np.ndarray):
        p_B_given_A_valid = p_B_given_A_valid.tolist()
    return _core.law_of_total_probability(p_B_given_A_valid, p_A_valid)

    @classmethod
    def sigmoid(cls, x: Union[Real, Sequence[Real]]) -> float:
        validated_input = cls._validate_input(_input=x, input_name="x", input_type=Real)
        return _core.sigmoid(float(validated_input))

    @classmethod
    def logit(cls, p: Real) -> float:
        validated = cls._validate_input(_input=p, input_name="p", input_type=Real)
        return _core.logit(float(validated))

    @classmethod
    def euclidean_distance(cls, x: Sequence[Real], y: Sequence[Real]) -> float:
        # Convert to list if numpy array; leave as-is if already Python list
        if hasattr(x, "tolist"):
            x = x.tolist()
        if hasattr(y, "tolist"):
            y = y.tolist()

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        return _core.euclidean_distance(x, y)

    @classmethod
    def manhattan_distance(cls, x: Sequence[Real], y: Sequence[Real]) -> float:
        if hasattr(x, "tolist"):
            x = x.tolist()
        if hasattr(y, "tolist"):
            y = y.tolist()

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        return _core.manhattan_distance(x, y)

    @classmethod
    def cosine_similarity(cls, x: Sequence[Real], y: Sequence[Real]) -> float:
        if hasattr(x, "tolist"):
            x = x.tolist()
        if hasattr(y, "tolist"):
            y = y.tolist()

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        return _core.cosine_similarity(x, y)

    @classmethod
    def coefficient_of_variation(cls, mean: Real, stddev: Real) -> float:
        cls._validate_input(_input=mean, input_name="mean", input_type=Real)
        cls._validate_input(_input=stddev, input_name="stddev", input_type=Real)
        return _core.coefficient_of_variation(float(mean), float(stddev))

    @classmethod
    def covariance(cls, mean_x: Real, mean_y: Real, E_xy: Real) -> float:
        cls._validate_input(_input=mean_x, input_name="mean_x", input_type=Real)
        cls._validate_input(_input=mean_y, input_name="mean_y", input_type=Real)
        cls._validate_input(_input=E_xy, input_name="E_xy", input_type=Real)
        return _core.covariance(float(mean_x), float(mean_y), float(E_xy))

    @classmethod
    def choose(cls, n: int, k: int) -> float:
        cls._validate_input(_input=n, input_name="n", input_type=int)
        cls._validate_input(_input=k, input_name="k", input_type=int)
        return _core.choose(n, k)

    @classmethod
    def permutation(cls, n: int, k: int) -> float:
        cls._validate_input(_input=n, input_name="n", input_type=int)
        cls._validate_input(_input=k, input_name="k", input_type=int)
        return _core.permutation(n, k)

    @classmethod
    def factorial(cls, n: int) -> float:
        cls._validate_input(_input=n, input_name="n", input_type=int)
        return _core.factorial(n)

    @classmethod
    def gamma(cls, x: Real) -> float:
        cls._validate_input(_input=x, input_name="x", input_type=Real)
        return _core.gamma(float(x))

    @classmethod
    def log_gamma(cls, x: Real) -> float:
        cls._validate_input(_input=x, input_name="x", input_type=Real)
        return _core.log_gamma(float(x))

    @classmethod
    def binomial(cls, n: int, a: Real, b: Real) -> float:
        cls._validate_input(_input=n, input_name="n", input_type=int)
        cls._validate_input(_input=a, input_name="a", input_type=Real)
        cls._validate_input(_input=b, input_name="b", input_type=Real)
        return _core.binomial(n, float(a), float(b))

    # --------------------
    # Batch Static Methods
    # --------------------
    @classmethod
    def sigmoid_cpu(cls, x: Sequence[Real]) -> NDArray[np.float64]:
        validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=1)
        return _core.sigmoid_cpu(validated)

    @classmethod
    def logit_cpu(cls, p: Sequence[Real]) -> NDArray[np.float64]:
        validated = cls._validate_input(_input=p, input_name="p", input_type=Sequence, dims=1)
        return _core.logit_cpu(validated)

    # -------------------
    # CUDA Static Methods
    # -------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def sigmoid_cuda(cls, x: Sequence[Real]) -> NDArray[np.float64]:
            validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=1)
            return _core.sigmoid_cuda(validated)

        @classmethod
        def logit_cuda(cls, p: Sequence[Real]) -> NDArray[np.float64]:
            validated = cls._validate_input(_input=p, input_name="p", input_type=Sequence, dims=1)
            return _core.logit_cuda(validated)

        @classmethod
        def euclidean_distance_cuda(cls, x: Sequence[Real], y: Sequence[Real]) -> NDArray[np.float64]:
            x_validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=2)
            y_validated = cls._validate_input(_input=y, input_name="y", input_type=Sequence, dims=2)

            if x_validated.shape != y_validated.shape:
                raise ValueError("x and y must have the same shape")

            return _core.euclidean_distance_cuda(x_validated, y_validated)

        @classmethod
        def manhattan_distance_cuda(cls, x: Sequence[Real], y: Sequence[Real]) -> NDArray[np.float64]:
            x_validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=2)
            y_validated = cls._validate_input(_input=y, input_name="y", input_type=Sequence, dims=2)

            if x_validated.shape != y_validated.shape:
                raise ValueError("x and y must have the same shape")

            return _core.manhattan_distance_cuda(x_validated, y_validated)

        @classmethod
        def cosine_similarity_cuda(cls, x: Sequence[Real], y: Sequence[Real]) -> NDArray[np.float64]:
            x_validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=2)
            y_validated = cls._validate_input(_input=y, input_name="y", input_type=Sequence, dims=2)

            if x_validated.shape != y_validated.shape:
                raise ValueError("x and y must have the same shape")

            return _core.cosine_similarity_cuda(x_validated, y_validated)
    else:
        @classmethod
        def sigmoid_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def logit_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def euclidean_distance_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def manhattan_distance_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def cosine_similarity_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )
