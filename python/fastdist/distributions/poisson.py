# python/fastdist/distributions/poisson.py
try:
    from .. import _fastdist as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "fastdist's compiled extension (_fastdist) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

from .. import config

from numbers import Real
from typing import SupportsFloat, Union, cast
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'poisson_pmf_cuda')


class Poisson:
    """
    Poisson distribution.

    Represents the number of events occurring in a fixed interval when events
    arrive independently at a constant average rate. Supports scalar and batch
    evaluation of PMF, CDF, MGF, CGF, and sampling. Methods automatically
    select CPU or CUDA backend if available and input is large.

        P(X = k) = lambda^k * exp(-lambda) / k!,  k = 0, 1, 2, ...

    Parameters
    ----------
    lambda_ : float
        Expected number of events in the interval (must be positive and
        finite). Named with a trailing underscore because `lambda` is a Python
        keyword.

    Attributes
    ----------
    lambda_ : float
        Expected number of events.

    Notes
    -----
    The mean and the variance are both `lambda_`. Count data whose variance
    exceeds its mean is overdispersed and is usually better served by
    `NegativeBinomial`.

    The moment methods accept an optional `lambda_` that overrides the
    instance's own, so they can be used for a one-off evaluation without
    constructing a second object.

    The PMF is evaluated in log space, so a large `lambda_` does not overflow
    even though ``lambda^k / k!`` would.

    This class is intended for high-performance computation using the
    underlying C++ core (`_fastdist`). Batch computations may use CPU or
    GPU (CUDA) depending on availability and input size.
    """

    # Magic Methods
    __slots__ = ("_lambda_",)

    def __init__(self, lambda_: SupportsFloat):
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
    def _validate_params(lambda_: SupportsFloat) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(lambda_, Real):
            raise TypeError("lambda_ must be a real number")
        if not math.isfinite(lambda_):
            raise ValueError("lambda_ must be finite")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

    @staticmethod
    def _validate_inputs(_input: Union[SupportsFloat, ArrayLike], input_name: str, step_size: Union[SupportsFloat, None] = None) \
            -> Union[float, np.ndarray]:
        """
        Validate inputs for Poisson methods.

        Ensures that scalars and sequences are the correct type and optionally
        checks that the input can be processed in batch.

        Parameters
        ----------
        _input : int, float, or sequence of ints/floats
            Input value(s) to validate.
        input_name : str
            Name of the input variable for error messages (`'x'` or `'t'`).
        step_size : float or None, optional
            Stride for subsampling batch inputs. Must be a real number.

        Returns
        -------
        int, float, or np.ndarray
            Validated scalar or 1D array of numbers.

        Raises
        ------
        TypeError
            If `_input` or `step_size` is of the wrong type.
        ValueError
            If `_input` array is not one-dimensional.
        """
        if _input is None:
            raise TypeError(f"{input_name} cannot be None")

        # numpy reads "0.5" as data, so a string would reach the array branch
        # and come back as a one-element result instead of a TypeError.
        if isinstance(_input, (str, bytes)):
            raise TypeError(f"{input_name} must be a real number or a sequence of them")

        # Declared up front: without it the type is inferred from the scalar
        # branch alone and the array branch looks like a bad assignment.
        validated: Union[float, np.ndarray]
        if isinstance(_input, Real):
            validated = float(_input)
        else:
            validated = Poisson._validate_array(arr=cast(ArrayLike, _input), input_name=input_name)
        if step_size is not None and not isinstance(step_size, int):
            raise TypeError("step_size must be an integer")

        return validated

    @staticmethod
    def _validate_array(arr: ArrayLike, input_name: str) -> np.ndarray:
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
    def pmf(self, x: Union[SupportsFloat, ArrayLike],
            step_size: int = 0) -> Union[float, NDArray]:
        """
        Compute the Poisson probability mass function (PMF).

        Evaluates P(X = x) at a single value or a sequence of values.
        Automatically chooses scalar, CPU batch, or CUDA batch computation
        depending on input size and availability.

        Parameters
        ----------
        x : float or Sequence[float]
            Value(s) at which to evaluate the PMF. Non-integer and negative
            values yield 0.
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            Probability mass at `x`, or array of probabilities if `x` is a
            sequence.

        Raises
        ------
        TypeError
            If input is not a real number or sequence of them.
        ValueError
            If input contains invalid values.

        Notes
        -----
        Evaluated in log space, so a large `lambda_` does not overflow even
        though ``lambda^k / k!`` would.
        """
        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)

        if not isinstance(validated_input, np.ndarray):
            # isinstance(..., np.ndarray) rather than numbers.Real so type
            # checkers can narrow the union _validate_inputs returns.
            return _core.poisson_pmf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_pmf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_pmf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_pmf_cpu(validated_input, self.lambda_, step_size)

    def cdf(self, x: Union[SupportsFloat, ArrayLike],
            step_size: int = 0) -> Union[float, np.ndarray]:
        """
        Compute the Poisson cumulative distribution function (CDF).

        Evaluates P(X <= x) at a single value or a sequence of values.
        Automatically chooses scalar, CPU batch, or CUDA batch computation
        depending on input size and availability.

        Parameters
        ----------
        x : float or Sequence[float]
            Value(s) at which to evaluate the CDF.
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            Cumulative probability at `x`, or array of probabilities if `x` is a
            sequence.

        Raises
        ------
        TypeError
            If input is not a real number or sequence of them.
        ValueError
            If input contains invalid values.
        """
        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)

        if not isinstance(validated_input, np.ndarray):
            return _core.poisson_cdf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_cdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cdf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_cdf_cpu(validated_input, self.lambda_, step_size)

    def mean(self, lambda_: Union[SupportsFloat, None] = None) -> float:
        """
        Compute the mean of the distribution.

        Parameters
        ----------
        lambda_ : float or None, optional
            Rate to use instead of the instance's own. Validated when given.

        Returns
        -------
        float
            `lambda_`.
        """
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_mean(lambda_)

    def variance(self, lambda_: Union[SupportsFloat, None] = None) -> float:
        """
        Compute the variance of the distribution.

        Parameters
        ----------
        lambda_ : float or None, optional
            Rate to use instead of the instance's own. Validated when given.

        Returns
        -------
        float
            `lambda_`, which equals the mean for this distribution.
        """
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_variance(lambda_)

    def stddev(self, lambda_: Union[SupportsFloat, None] = None) -> float:
        """
        Compute the standard deviation of the distribution.

        Parameters
        ----------
        lambda_ : float or None, optional
            Rate to use instead of the instance's own. Validated when given.

        Returns
        -------
        float
            ``sqrt(lambda_)``.
        """
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_stddev(lambda_)

    def mgf(self, t: Union[SupportsFloat, ArrayLike],
            step_size: int = 0) -> Union[float, np.ndarray]:
        """
        Compute the Poisson moment generating function (MGF).

        Parameters
        ----------
        t : float or Sequence[float]
            Point(s) at which to evaluate the MGF.
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            ``exp(lambda_ * (e^t - 1))``, or an array of them.

        Raises
        ------
        TypeError
            If input is not a real number or sequence of them.

        Notes
        -----
        Overflows to inf for a moderate `t`; use `cgf` there.
        """
        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if not isinstance(validated_input, np.ndarray):
            return _core.poisson_mgf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_mgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_mgf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_mgf_cpu(validated_input, self.lambda_, step_size)

    def cgf(self, t: Union[SupportsFloat, ArrayLike],
            step_size: int = 0) -> Union[float, np.ndarray]:
        """
        Compute the Poisson cumulant generating function (CGF).

        Parameters
        ----------
        t : float or Sequence[float]
            Point(s) at which to evaluate the CGF.
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            ``lambda_ * (e^t - 1)``, or an array of them.

        Raises
        ------
        TypeError
            If input is not a real number or sequence of them.

        Notes
        -----
        Computed directly rather than as log(MGF), so it stays finite well past
        the point where `mgf` overflows.
        """
        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)

        if not isinstance(validated_input, np.ndarray):
            return _core.poisson_cgf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("poisson_cgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cgf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.poisson_cgf_cpu(validated_input, self.lambda_, step_size)

    def sample(self, lambda_: Union[SupportsFloat, None] = None) -> int:
        """
        Draw a random sample from the distribution.

        Parameters
        ----------
        lambda_ : float or None, optional
            Rate to use instead of the instance's own. Validated when given.

        Returns
        -------
        int
            One draw from Poisson(`lambda_`).

        Notes
        -----
        Seed the shared engine with `fastdist.seed` for reproducible draws.
        """
        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.poisson_sample(lambda_)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, x: SupportsFloat, lambda_: SupportsFloat) -> float:
        """
        Compute the PMF for a single value without constructing an instance.

        Parameters
        ----------
        x : float
            Point at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).

        Returns
        -------
        float
            Probability mass at `x`.

        Raises
        ------
        TypeError
            If an argument is not a real number.
        ValueError
            If `lambda_` is not positive or not finite.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.poisson_pmf_scalar(float(x), float(lambda_))

    @classmethod
    def _cdf_scalar(cls, x: SupportsFloat, lambda_: SupportsFloat) -> float:
        """
        Compute the CDF for a single value without constructing an instance.

        Parameters
        ----------
        x : float
            Point at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).

        Returns
        -------
        float
            Cumulative probability at `x`.

        Raises
        ------
        TypeError
            If an argument is not a real number.
        ValueError
            If `lambda_` is not positive or not finite.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.poisson_cdf_scalar(float(x), float(lambda_))

    @classmethod
    def _mgf_scalar(cls, t: SupportsFloat, lambda_: SupportsFloat) -> float:
        """
        Compute the MGF for a single value without constructing an instance.

        Parameters
        ----------
        t : float
            Point at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).

        Returns
        -------
        float
            ``exp(lambda_ * (e^t - 1))``.

        Raises
        ------
        TypeError
            If an argument is not a real number.
        ValueError
            If `lambda_` is not positive or not finite.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.poisson_mgf_scalar(float(t), float(lambda_))

    @classmethod
    def _cgf_scalar(cls, t: SupportsFloat, lambda_: SupportsFloat) -> float:
        """
        Compute the CGF for a single value without constructing an instance.

        Parameters
        ----------
        t : float
            Point at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).

        Returns
        -------
        float
            ``lambda_ * (e^t - 1)``.

        Raises
        ------
        TypeError
            If an argument is not a real number.
        ValueError
            If `lambda_` is not positive or not finite.

        Notes
        -----
        Parameters are validated on every call, so this form is slower than
        building an instance when evaluating repeatedly.
        """
        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.poisson_cgf_scalar(float(t), float(lambda_))

    # ------------------------------------------------------------------------------------------------------------------
    # Batch Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_cpu(cls, x: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
        """
        Compute the PMF over an array on the CPU.

        Parameters
        ----------
        x : array-like of floats
            Values at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        np.ndarray
            One value per element of `x`.

        Raises
        ------

        TypeError
            If an argument is of the wrong type.
        ValueError
            If `lambda_` is not positive, or the input is not one-dimensional.
        """
        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_pmf_cpu(x, lambda_, step_size)

    @classmethod
    def _cdf_cpu(cls, x: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
        """
        Compute the CDF over an array on the CPU.

        Parameters
        ----------
        x : array-like of floats
            Values at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        np.ndarray
            One value per element of `x`.

        Raises
        ------

        TypeError
            If an argument is of the wrong type.
        ValueError
            If `lambda_` is not positive, or the input is not one-dimensional.
        """
        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_cdf_cpu(x, lambda_, step_size)

    @classmethod
    def _mgf_cpu(cls, t: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
        """
        Compute the MGF over an array on the CPU.

        Parameters
        ----------
        t : array-like of floats
            Values at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        np.ndarray
            One value per element of `t`.

        Raises
        ------

        TypeError
            If an argument is of the wrong type.
        ValueError
            If `lambda_` is not positive, or the input is not one-dimensional.
        """
        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_mgf_cpu(t, lambda_, step_size)

    @classmethod
    def _cgf_cpu(cls, t: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
        """
        Compute the CGF over an array on the CPU.

        Parameters
        ----------
        t : array-like of floats
            Values at which to evaluate.
        lambda_ : float
            Rate parameter (must be positive and finite).
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        np.ndarray
            One value per element of `t`.

        Raises
        ------

        TypeError
            If an argument is of the wrong type.
        ValueError
            If `lambda_` is not positive, or the input is not one-dimensional.
        """
        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.poisson_cgf_cpu(t, lambda_, step_size)

    # ------------------------------------------------------------------------------------------------------------------
    # CUDA Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def _pmf_cuda(cls, x: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
            """
            Compute the PMF over an array on the GPU.

            Parameters
            ----------
            x : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `x`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            cls._validate_params(lambda_=lambda_)
            validated_input = cast(np.ndarray, cls._validate_inputs(_input=x, input_name="x", step_size=step_size))
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_pmf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _cdf_cuda(cls, x: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
            """
            Compute the CDF over an array on the GPU.

            Parameters
            ----------
            x : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `x`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            cls._validate_params(lambda_=lambda_)
            validated_input = cast(np.ndarray, cls._validate_inputs(_input=x, input_name="x", step_size=step_size))
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cdf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _mgf_cuda(cls, t: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
            """
            Compute the MGF over an array on the GPU.

            Parameters
            ----------
            t : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `t`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            cls._validate_params(lambda_=lambda_)
            validated_input = cast(np.ndarray, cls._validate_inputs(_input=t, input_name="t", step_size=step_size))
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_mgf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _cgf_cuda(cls, t: ArrayLike, lambda_: SupportsFloat, step_size: int = 0) -> NDArray[np.float64]:
            """
            Compute the CGF over an array on the GPU.

            Parameters
            ----------
            t : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `t`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            cls._validate_params(lambda_=lambda_)
            validated_input = cast(np.ndarray, cls._validate_inputs(_input=t, input_name="t", step_size=step_size))
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.poisson_cgf_cuda(validated_input, lambda_, step_size)
    else:
        @classmethod
        def _pmf_cuda(cls, *args, **kwargs):
            """
            Compute the PMF over an array on the GPU.

            Parameters
            ----------
            x : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `x`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _cdf_cuda(cls, *args, **kwargs):
            """
            Compute the CDF over an array on the GPU.

            Parameters
            ----------
            x : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `x`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _mgf_cuda(cls, *args, **kwargs):
            """
            Compute the MGF over an array on the GPU.

            Parameters
            ----------
            t : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `t`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _cgf_cuda(cls, *args, **kwargs):
            """
            Compute the CGF over an array on the GPU.

            Parameters
            ----------
            t : array-like of floats
                Values at which to evaluate.
            lambda_ : float
                Rate parameter (must be positive and finite).
            step_size : int, optional
                Stride used to subsample the sequence. Default is 0 (no subsampling).

            Returns
            -------
            np.ndarray
                One value per element of `t`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If an argument is of the wrong type.
            ValueError
                If `lambda_` is not positive, or the input is not one-dimensional.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )
