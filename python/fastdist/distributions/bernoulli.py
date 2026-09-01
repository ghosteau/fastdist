# python/distributions/bernoulli.py
try:
    from fastdist import _fastdist as _core
    from fastdist import config
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import numpy as np
from numbers import Real
from typing import Sequence, Union
from numpy.typing import NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'bernoulli_pmf_cuda')


class Bernoulli:
    """
    Bernoulli distribution.

    Represents a Bernoulli-distributed random variable, which takes value 1
    with probability `p` and value 0 with probability `1 - p`. Supports
    scalar and batch evaluation of PMF, CDF, MGF, CGF, and sampling. Methods
    automatically select CPU or CUDA backend if available and input is large.

    Parameters
    ----------
    p : float
        Probability of success (must be in [0, 1]).

    Attributes
    ----------
    p : float
        Probability of success.

    Notes
    -----
    This class is intended for high-performance computation using the
    underlying C++ core (`_fastdist`). Batch computations may use CPU or
    GPU (CUDA) depending on availability and input size.
    """

    # Magic Methods
    __slots__ = ("_p",)

    def __init__(self, p: Real):
        """
        Initialize a Bernoulli distribution instance.

        Parameters
        ----------
        p : float
            Probability of success (must be in [0, 1]).

        Raises
        ------
        TypeError
            If `p` is not a real number.
        ValueError
            If `p` is outside [0, 1].
        """

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
    def _validate_params(p: Real) -> None:
        """
        Validate the probability parameter `p`.

        Parameters
        ----------
        p : float
            Probability of success.

        Raises
        ------
        TypeError
            If `p` is not a real number.
        ValueError
            If `p` is outside [0, 1].

        Notes
        -----
        Used internally to ensure all methods receive a valid probability.
        """

        if not isinstance(p, Real):
            raise TypeError("p must be a real number")
        if p < 0 or p > 1:
            raise ValueError("p must be in the interval [0, 1]")

    @staticmethod
    def _validate_inputs(_input: Union[int, Real, Sequence[int], Sequence[Real]], input_name: str,
                         step_size: Union[Real, None] = None) -> Union[
        int, Real, NDArray[np.int64], NDArray[np.float64]]:
        """
        Validate inputs for Bernoulli methods.

        Ensures that scalars and sequences are the correct type and optionally
        checks that the input can be processed in batch.

        Parameters
        ----------
        _input : int, float, or sequence of ints/floats
            Input value(s) to validate.
        input_name : str
            Name of the input variable for error messages (`'k'` or `'t'`).
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
            raise TypeError(f"{input_name} must not be None")

        # Scalar input
        if isinstance(_input, Real):
            if input_name == "k":
                if not isinstance(_input, int):
                    raise TypeError(f"{input_name} must be an integer")
                else:
                    validated = int(_input)
            elif input_name == "t":
                if not isinstance(_input, Real):
                    raise TypeError(f"{input_name} must be a real number")
                else:
                    validated = float(_input)
            else:
                raise ValueError(f"Unknown input_name: {input_name}")

        # Sequence Input
        else:
            validated = Bernoulli._validate_array(arr=_input, input_name=input_name)

            if input_name == "k":
                if not np.issubdtype(validated.dtype, np.integer):
                    raise TypeError(f"{input_name} must be an integer or array-like of integers")
            elif input_name == "t":
                if not np.issubdtype(validated.dtype, np.number):
                    raise TypeError(f"{input_name} must be a real number or array-like of real numbers")

        if step_size is not None and not isinstance(step_size, Real):
            raise TypeError("step_size must be a real number")

        return validated

    @staticmethod
    def _validate_array(arr: Sequence[Real], input_name: str) -> NDArray[np.int64]:
        """
        Validate and convert a sequence to a NumPy array.

        Parameters
        ----------
        arr : sequence of numbers
            Input sequence to validate and convert.
        input_name : str
            Name of the input variable for error messages.

        Returns
        -------
        np.ndarray
            1D array of integers.

        Raises
        ------
        TypeError
            If the input sequence contains non-numeric elements.
        ValueError
            If the input sequence is not one-dimensional.

        Notes
        -----
        This method is used internally by `_validate_inputs` to ensure all
        sequences are safely converted to NumPy arrays before computation.
        """

        try:
            arr = np.atleast_1d(arr).astype(np.int64)
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
    def pmf(self, k: Union[int, Sequence[int]], step_size: int = 0) -> Union[Real, np.ndarray]:
        """
        Compute the Bernoulli probability mass function (PMF).

        Evaluates the PMF P(X = k) for a Bernoulli-distributed random variable
        at a single value or a sequence of values. Automatically chooses scalar,
        CPU batch, or CUDA batch computation depending on input size and availability.

        Parameters
        ----------
        k : int or Sequence[int]
            Value(s) at which to evaluate the PMF. Must be 0 or 1.
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            Probability mass at `k`, or array of probabilities if `k` is a sequence.

        Raises
        ------
        TypeError
            If input is not an int or sequence of ints.
        ValueError
            If input contains invalid values.
        """

        validated_input = self._validate_inputs(_input=k, input_name="k", step_size=step_size)
        if isinstance(validated_input, int):
            return _core.bernoulli_pmf_scalar(validated_input, self.p)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("bernoulli_pmf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_pmf_cuda(validated_input, self.p, step_size)
        else:
            return _core.bernoulli_pmf_cpu(validated_input, self.p, step_size)

    def cdf(self, k: Union[int, Sequence[int]], step_size: int = 0) -> Union[Real, np.ndarray]:
        """
        Compute the Bernoulli cumulative distribution function (CDF).

        Evaluates the CDF P(X ≤ k) at a single value or sequence of values. Automatically
        chooses scalar, CPU batch, or CUDA batch computation.

        Parameters
        ----------
        k : int or Sequence[int]
            Value(s) at which to evaluate the CDF. Must be 0 or 1.
        step_size : int, optional
            Stride used to subsample the sequence. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            Cumulative probability(s) for each value of `k`.

        Raises
        ------
        TypeError
            If input is not an int or sequence of ints.
        ValueError
            If input contains invalid values.
        """

        validated_input = self._validate_inputs(_input=k, input_name="k", step_size=step_size)
        if isinstance(validated_input, int):
            return _core.bernoulli_cdf_scalar(validated_input, self.p)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("bernoulli_cdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_cdf_cuda(validated_input, self.p, step_size)
        else:
            return _core.bernoulli_cdf_cpu(validated_input, self.p, step_size)

    def mean(self, p: Union[Real, None] = None) -> Real:
        """
        Compute the mean of the Bernoulli distribution.

        Parameters
        ----------
        p : float, optional
            Probability of success. Defaults to the instance's `p`.

        Returns
        -------
        float
            The mean (expected value) of the Bernoulli distribution: `p`.

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        """

        if p is None:
            p = self.p
        else:
            self._validate_params(p=p)
        return _core.bernoulli_mean(p)

    def variance(self, p: Union[Real, None] = None) -> Real:
        """
        Compute the variance of the Bernoulli distribution.

        Parameters
        ----------
        p : float, optional
            Probability of success. Defaults to the instance's `p`.

        Returns
        -------
        float
            The variance of the Bernoulli distribution: `p * (1 - p)`.

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        """

        if p is None:
            p = self.p
        else:
            self._validate_params(p=p)
        return _core.bernoulli_variance(p)

    def stddev(self, p: Union[Real, None] = None) -> Real:
        """
        Compute the standard deviation of the Bernoulli distribution.

        Parameters
        ----------
        p : float, optional
            Probability of success. Defaults to the instance's `p`.

        Returns
        -------
        float
            The standard deviation: sqrt(p * (1 - p)).

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        """

        if p is None:
            p = self.p
        else:
            self._validate_params(p=p)
        return _core.bernoulli_stddev(p)

    def mgf(self, t: Union[Real, Sequence[Real]],
            step_size: int = 0) -> Union[Real, np.ndarray]:
        """
        Compute the moment-generating function (MGF) of the Bernoulli distribution.

        Parameters
        ----------
        t : float or Sequence[float]
            Point(s) at which to evaluate the MGF.
        step_size : int, optional
            Stride used to subsample sequences. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            MGF evaluated at `t`.

        Raises
        ------
        TypeError
            If `t` is not a real number or sequence of real numbers.
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.bernoulli_mgf_scalar(validated_input, self.p)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("bernoulli_mgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_mgf_cuda(validated_input, self.p, step_size)
        else:
            return _core.bernoulli_mgf_cpu(validated_input, self.p, step_size)

    def cgf(self, t: Union[Real, Sequence[Real]], step_size: int = 0) -> Union[Real, np.ndarray]:
        """
        Compute the cumulant-generating function (CGF) of the Bernoulli distribution.

        Parameters
        ----------
        t : float or Sequence[float]
            Point(s) at which to evaluate the CGF.
        step_size : int, optional
            Stride used to subsample sequences. Default is 0 (no subsampling).

        Returns
        -------
        float or ndarray
            CGF evaluated at `t`.

        Raises
        ------
        TypeError
            If `t` is not a real number or sequence of real numbers.
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.bernoulli_cgf_scalar(validated_input, self.p)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("bernoulli_cgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_cgf_cuda(validated_input, self.p, step_size)
        else:
            return _core.bernoulli_cgf_cpu(validated_input, self.p, step_size)

    def sample(self, p: Union[Real, None] = None) -> int:
        """
        Draw a single random sample from the Bernoulli distribution.

        Parameters
        ----------
        p : float, optional
            Probability of success. Defaults to the instance's `p`.

        Returns
        -------
        int
            0 or 1, sampled according to `p`.

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        """

        if p is None:
            p = self.p
        else:
            self._validate_params(p=p)
        return _core.bernoulli_sample(p)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, k: int, p: Real) -> Real:
        """
        Compute the PMF for a single scalar value.

        Parameters
        ----------
        k : int
            The value at which to evaluate the PMF.
        p : float
            Probability of success.

        Returns
        -------
        float
            Probability P(X = k).

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        TypeError
            If `k` is not an integer.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=k, input_name="k")
        return _core.bernoulli_pmf_scalar(k, float(p))

    @classmethod
    def _cdf_scalar(cls, k: int, p: Real) -> Real:
        """
        Compute the CDF for a single scalar value.

        Parameters
        ----------
        k : int
            The value at which to evaluate the CDF.
        p : float
            Probability of success.

        Returns
        -------
        float
            Cumulative probability P(X ≤ k).

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        TypeError
            If `k` is not an integer.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=k, input_name="k")
        return _core.bernoulli_cdf_scalar(k, float(p))

    @classmethod
    def _mgf_scalar(cls, t: Real, p: Real) -> Real:
        """
        Compute the MGF for a single scalar point.

        Parameters
        ----------
        t : float
            Point at which to evaluate the MGF.
        p : float
            Probability of success.

        Returns
        -------
        float
            Moment-generating function evaluated at `t`.

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        TypeError
            If `t` is not a real number.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.bernoulli_mgf_scalar(float(t), float(p))

    @classmethod
    def _cgf_scalar(cls, t: Real, p: Real) -> Real:
        """
        Compute the CGF for a single scalar point.

        Parameters
        ----------
        t : float
            Point at which to evaluate the CGF.
        p : float
            Probability of success.

        Returns
        -------
        float
            Cumulant-generating function evaluated at `t`.

        Raises
        ------
        ValueError
            If `p` is outside [0, 1].
        TypeError
            If `t` is not a real number.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.bernoulli_cgf_scalar(float(t), float(p))

    # ------------------------------------------------------------------------------------------------------------------
    # Batch Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_cpu(cls, k: Sequence[int], p: Real, step_size: int = 0) -> NDArray[np.float64]:
        """
        Bernoulli probability mass function (CPU, batch).

        Computes the probability mass function (PMF) of a Bernoulli-distributed
        random variable for a batch of input values using the CPU backend.

        Parameters
        ----------
        k : Sequence[int]
            Sequence of integer values at which to evaluate the PMF.
        step_size : int, optional
            Stride used to subsample the input sequence. A value of 0 disables
            subsampling.

        Returns
        -------
        NDArray[np.float64]
            Array of PMF values corresponding to each input in `k`.

        Notes
        -----
        The PMF is defined as:
            P(X = k) = p^k (1 - p)^(1 - k), for k ∈ {0, 1}

        Input validation is performed prior to execution.

        Raises
        ------
        ValueError
            If input values are invalid or incompatible with the Bernoulli
            distribution.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=k, input_name="k", step_size=step_size)
        return _core.bernoulli_pmf_cpu(k, p, step_size)

    @classmethod
    def _cdf_cpu(cls, k: Sequence[int], p: Real, step_size: int = 0) -> NDArray[np.float64]:
        """
        Bernoulli cumulative distribution function (CPU, batch).

        Computes the cumulative distribution function (CDF) of a Bernoulli-
        distributed random variable for a batch of input values using the CPU
        backend.

        Parameters
        ----------
        k : Sequence[int]
            Sequence of integer values at which to evaluate the CDF.
        step_size : int, optional
            Stride used to subsample the input sequence. A value of 0 disables
            subsampling.

        Returns
        -------
        NDArray[np.float64]
            Array of CDF values corresponding to each input in `k`.

        Notes
        -----
        The CDF is defined as:
            P(X ≤ k)

        Input validation is performed prior to execution.

        Raises
        ------
        ValueError
            If input values are invalid or incompatible with the Bernoulli
            distribution.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=k, input_name="k", step_size=step_size)
        return _core.bernoulli_cdf_cpu(k, p, step_size)

    @classmethod
    def _mgf_cpu(cls, t: Sequence[Real], p: Real, step_size: int = 0) -> NDArray[np.float64]:
        """
        Bernoulli moment-generating function (CPU, batch).

        Computes the moment-generating function (MGF) of a Bernoulli-distributed
        random variable for a batch of input values using the CPU backend.

        Parameters
        ----------
        t : Sequence[float]
            Sequence of real values at which to evaluate the MGF.
        step_size : int, optional
            Stride used to subsample the input sequence. A value of 0 disables
            subsampling.

        Returns
        -------
        NDArray[np.float64]
            Array of MGF values corresponding to each input in `t`.

        Notes
        -----
        The MGF is defined as:
            M(t) = (1 - p) + p * exp(t)

        Input validation is performed prior to execution.

        Raises
        ------
        ValueError
            If input values are invalid.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        return _core.bernoulli_mgf_cpu(t, p, step_size)

    @classmethod
    def _cgf_cpu(cls, t: Sequence[Real], p: Real, step_size: int = 0) -> NDArray[np.float64]:
        """
        Bernoulli cumulant-generating function (CPU, batch).

        Computes the cumulant-generating function (CGF) of a Bernoulli-distributed
        random variable for a batch of input values using the CPU backend.

        Parameters
        ----------
        t : Sequence[float]
            Sequence of real values at which to evaluate the CGF.
        step_size : int, optional
            Stride used to subsample the input sequence. A value of 0 disables
            subsampling.

        Returns
        -------
        NDArray[np.float64]
            Array of CGF values corresponding to each input in `t`.

        Notes
        -----
        The CGF is defined as:
            K(t) = log((1 - p) + p * exp(t))

        Input validation is performed prior to execution.

        Raises
        ------
        ValueError
            If input values are invalid.
        """

        cls._validate_params(p=p)
        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        return _core.bernoulli_cgf_cpu(t, p, step_size)

    # ------------------------------------------------------------------------------------------------------------------
    # CUDA Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def _pmf_cuda(cls, k: Sequence[int], p: Real, step_size: int = 0) -> NDArray[np.float64]:
            """
            Bernoulli probability mass function (CUDA, batch).

            Computes the probability mass function (PMF) of a Bernoulli-distributed
            random variable for a batch of input values using the CUDA backend.

            Parameters
            ----------
            k : Sequence[int]
                Sequence of integer values at which to evaluate the PMF.
            step_size : int, optional
                Stride used to subsample the input sequence. A value of 0 disables
                subsampling.

            Returns
            -------
            NDArray[np.float64]
                Array of PMF values corresponding to each input in `k`.

            Notes
            -----
            This method requires CUDA support and an available GPU.

            Raises
            ------
            RuntimeError
                If CUDA is unavailable or improperly configured.
            """

            cls._validate_params(p=p)
            validated_input = cls._validate_inputs(_input=k, input_name="k", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_pmf_cuda(k=validated_input, p=p, step_size=step_size)

        @classmethod
        def _cdf_cuda(cls, k: Sequence[int], p: Real, step_size: int = 0) -> NDArray[np.float64]:
            """
            Bernoulli cumulative distribution function (CUDA, batch).

            Computes the cumulative distribution function (CDF) of a Bernoulli-
            distributed random variable for a batch of input values using the CUDA
            backend.

            Parameters
            ----------
            k : Sequence[int]
                Sequence of integer values at which to evaluate the CDF.
            step_size : int, optional
                Stride used to subsample the input sequence. A value of 0 disables
                subsampling.

            Returns
            -------
            NDArray[np.float64]
                Array of CDF values corresponding to each input in `k`.

            Raises
            ------
            RuntimeError
                If CUDA is unavailable or improperly configured.
            """

            cls._validate_params(p=p)
            validated_input = cls._validate_inputs(_input=k, input_name="k", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_cdf_cuda(validated_input, p, step_size)

        @classmethod
        def _mgf_cuda(cls, t: Sequence[Real], p: Real, step_size: int = 0) -> NDArray[np.float64]:
            """
            Bernoulli moment-generating function (CUDA, batch).

            Computes the moment-generating function (MGF) of a Bernoulli-distributed
            random variable for a batch of input values using the CUDA backend.

            Parameters
            ----------
            t : Sequence[float]
                Sequence of real values at which to evaluate the MGF.
            step_size : int, optional
                Stride used to subsample the input sequence. A value of 0 disables
                subsampling.

            Returns
            -------
            NDArray[np.float64]
                Array of MGF values corresponding to each input in `t`.

            Raises
            ------
            RuntimeError
                If CUDA is unavailable or improperly configured.
            """

            cls._validate_params(p=p)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_mgf_cuda(validated_input, p, step_size)

        @classmethod
        def _cgf_cuda(cls, t: Sequence[Real], p: Real, step_size: int = 0) -> NDArray[np.float64]:
            """
            Bernoulli cumulant-generating function (CUDA, batch).

            Computes the cumulant-generating function (CGF) of a Bernoulli-distributed
            random variable for a batch of input values using the CUDA backend.

            Parameters
            ----------
            t : Sequence[float]
                Sequence of real values at which to evaluate the CGF.
            step_size : int, optional
                Stride used to subsample the input sequence. A value of 0 disables
                subsampling.

            Returns
            -------
            NDArray[np.float64]
                Array of CGF values corresponding to each input in `t`.

            Raises
            ------
            RuntimeError
                If CUDA is unavailable or improperly configured.
            """

            cls._validate_params(p=p)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.bernoulli_cgf_cuda(validated_input, p, step_size)
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
