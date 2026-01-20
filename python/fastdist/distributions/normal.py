# python/distributions/normal.py
try:
    from fastdist import _fastdist as _core
    from fastdist import config
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import math
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'normal_pdf_cuda')


class Normal:
    """
    Normal (Gaussian) distribution.

    Represents a normal distribution with mean mu and standard deviation sigma.
    Supports both scalar and vectorized operations with automatic CPU/CUDA
    acceleration for large arrays.

    Parameters
    ----------
    mu : int or float
        Mean (location parameter) of the distribution. Must be finite.
    sigma : int or float
        Standard deviation (scale parameter) of the distribution.
        Must be positive and finite.

    Attributes
    ----------
    mu : float
        The mean of the distribution.
    sigma : float
        The standard deviation of the distribution.

    Examples
    --------
    > dist = Normal(mu=0, sigma=1)
    > dist.pdf(0)
    0.3989422804014327
    > dist.cdf([−1, 0, 1])
    array([0.15865525, 0.5, 0.84134475])
    """
    # Magic Methods
    __slots__ = ("_mu", "_sigma")

    def __init__(self, mu: Union[int, float], sigma: Union[int, float]):
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
    def mu(self, value: Union[int, float]):
        self._validate_params(mu=value)
        self._mu = float(value)

    @sigma.setter
    def sigma(self, value: Union[int, float]):
        self._validate_params(sigma=value)
        self._sigma = float(value)

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    @staticmethod
    def _validate_params(mu: Union[int, float] = None,
                         sigma: Union[int, float] = None):
        """
        Validate the distribution parameters `mu` and `sigma`.

        Parameters
        ----------
        mu : int or float, optional
            Mean of the distribution. Must be finite if provided.
        sigma : int or float, optional
            Standard deviation of the distribution. Must be positive and finite if provided.

        Raises
        ------
        TypeError
            If `mu` or `sigma` is not a real number.
        ValueError
            If `mu` is not finite, or `sigma` is not finite or not positive.

        Notes
        -----
        This method is used internally by the class to ensure all distribution parameters
        are valid before performing calculations.
        """

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
    def _validate_inputs(_input: Union[int, float] | Sequence[Union[int, float]] = None, input_name: str = None,
                         step_size: Union[int, float, Sequence[Union[int, float]]] = None) -> Union[
        int, float, np.ndarray, None]:
        """
        Validate input values for distribution methods.

        Parameters
        ----------
        _input : int, float, or sequence of int/float, optional
            The input value(s) to validate.
        input_name : str, optional
            Name of the input variable (used in error messages).
        step_size : int, float, or sequence of int/float, optional
            Step size for vectorized operations. Must be a real number if provided.

        Returns
        -------
        int, float, np.ndarray, or None
            The validated input. Scalars are returned as-is, sequences are converted
            to a 1D NumPy array of type float64. Returns `None` if `_input` is `None`.

        Raises
        ------
        TypeError
            If `_input` or `step_size` has an invalid type.

        Notes
        -----
        This method is used internally to standardize inputs before passing them to
        scalar, CPU, or CUDA implementations.
        """

        validated = None
        if _input is not None:
            if isinstance(_input, (int, float)):
                validated = _input
            else:
                validated = Normal._validate_array(arr=_input, input_name=input_name)
        if step_size is not None and not isinstance(step_size, (int, float)):
            raise TypeError("step_size must be a real number")

        return validated

    @staticmethod
    def _validate_array(arr: Sequence[Union[int, float]], input_name: str) -> np.ndarray:
        """
        Convert a sequence of numbers to a validated 1D NumPy array.

        Parameters
        ----------
        arr : sequence of int or float
            Input array-like to validate and convert.
        input_name : str
            Name of the input variable (used in error messages).

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
        Used internally by `_validate_inputs` to standardize array-like inputs for
        CPU and CUDA operations.
        """

        try:
            arr = np.atleast_1d(arr).astype(np.float64)
        except (ValueError, TypeError):
            raise TypeError(f"{input_name} must be numeric (int, float, or array-like of numbers)")

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
        """
        return _CUDA_AVAILABLE

    # ----------------
    # Instance Methods
    # ----------------
    def pdf(self, x: Union[int, float, Sequence[Union[int, float]]],
            step_size: Union[int, float] = 0) -> float | np.ndarray:
        """
        Probability density function.

        Compute the value of the probability density function at x.

        Parameters
        ----------
        x : int, float, or array-like
            Point(s) at which to evaluate the PDF.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero, indicates that
            input values are evenly spaced, enabling computational shortcuts.

        Returns
        -------
        float or ndarray
            PDF value(s) at x. Returns float for scalar input,
            ndarray for array-like input.

        Examples
        --------
        > dist = Normal(mu=0, sigma=1)
        > dist.pdf(0)
        0.3989422804014327
        > dist.pdf([−1, 0, 1])
        array([0.24197072, 0.39894228, 0.24197072])

        Notes
        -----
        For large arrays (above the CUDA threshold), computation automatically
        uses GPU acceleration if available.
        """

        validated_input = self._validate_inputs(_input=x, input_name="x")
        if isinstance(validated_input, (int, float)):
            return _core.normal_pdf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_pdf"):
            return _core.normal_pdf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_pdf_cpu(validated_input, self.mu, self.sigma, step_size)

    def logpdf(self, x: Union[int, float, Sequence[Union[int, float]]],
               step_size: Union[int, float] = 0) -> float | np.ndarray:
        """
        Log probability density function.

        Compute the natural logarithm of the probability density function at x.
        More numerically stable than log(pdf(x)) for extreme values.

        Parameters
        ----------
        x : int, float, or array-like
            Point(s) at which to evaluate the log PDF.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero, indicates that
            input values are evenly spaced, enabling computational shortcuts.

        Returns
        -------
        float or ndarray
            Log PDF value(s) at x. Returns float for scalar input,
            ndarray for array-like input.

        Examples
        --------
        > dist = Normal(mu=0, sigma=1)
        > dist.logpdf(0)
        -0.9189385332046727
        > dist.logpdf([0, 1, 2])
        array([-0.91893853, -1.41893853, -2.91893853])

        Notes
        -----
        For large arrays (above the CUDA threshold), computation automatically
        uses GPU acceleration if available.
        """

        validated_input = self._validate_inputs(_input=x, input_name="x")
        if isinstance(validated_input, (int, float)):
            return _core.normal_logpdf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_logpdf"):
            return _core.normal_logpdf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_logpdf_cpu(validated_input, self.mu, self.sigma, step_size)

    def cdf(self, x: Union[int, float, Sequence[Union[int, float]]],
            step_size: Union[int, float] = 0) -> float | np.ndarray:
        """
        Cumulative distribution function.

        Compute the probability that a random variable X from this
        distribution is less than or equal to x.

        Parameters
        ----------
        x : int, float, or array-like
            Point(s) at which to evaluate the CDF.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero, indicates that
            input values are evenly spaced, enabling computational shortcuts.

        Returns
        -------
        float or ndarray
            CDF value(s) at x, in range [0, 1]. Returns float for scalar input,
            ndarray for array-like input.

        Examples
        --------
        > dist = Normal(mu=0, sigma=1)
        > dist.cdf(0)
        0.5
        > dist.cdf([−1, 0, 1])
        array([0.15865525, 0.5, 0.84134475])

        Notes
        -----
        For large arrays (above the CUDA threshold), computation automatically
        uses GPU acceleration if available.
        """

        validated_input = self._validate_inputs(_input=x, input_name="x")
        if isinstance(validated_input, (int, float)):
            return _core.normal_cdf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_cdf"):
            return _core.normal_cdf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_cdf_cpu(validated_input, self.mu, self.sigma, step_size)

    def mean(self) -> float:
        """
        Mean (expected value) of the distribution.

        Returns
        -------
        float
            The mean `mu` of the distribution.
        """

        return _core.normal_mean(self.mu)

    def variance(self) -> float:
        """
        Variance of the distribution.

        Returns
        -------
        float
            Variance of the distribution, equal to sigma squared.
        """
        return _core.normal_variance(self.sigma)

    def stddev(self) -> float:
        """
        Standard deviation of the distribution.

        Returns
        -------
        float
            Standard deviation, equal to `sigma`.
        """
        return _core.normal_stddev(self.sigma)

    def mgf(self, t: Union[int, float, Sequence[Union[int, float]]],
            step_size: Union[int, float] = 0) -> float | np.ndarray:
        """
        Moment generating function.

        Compute the moment generating function M(t) = E[exp(tX)].

        Parameters
        ----------
        t : int, float, or array-like
            Point(s) at which to evaluate the MGF.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero, indicates that
            input values are evenly spaced, enabling computational shortcuts.

        Returns
        -------
        float or ndarray
            MGF value(s) at t. Returns float for scalar input,
            ndarray for array-like input.

        Examples
        --------
        > dist = Normal(mu=0, sigma=1)
        > dist.mgf(0)
        1.0
        > dist.mgf([0, 0.5, 1])
        array([1., 1.13315, 1.64872])

        Notes
        -----
        For a normal distribution with mean mu and standard deviation sigma:

        . math::
            M(t) = \\exp\\left(\\mu t + \\frac{\\sigma^2 t^2}{2}\\right)

        For large arrays (above the CUDA threshold), computation automatically
        uses GPU acceleration if available.
        """

        validated_input = self._validate_inputs(_input=t, input_name="t")
        if isinstance(validated_input, (int, float)):
            return _core.normal_mgf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_mgf"):
            return _core.normal_mgf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_mgf_cpu(validated_input, self.mu, self.sigma, step_size)

    def cgf(self, t: Union[int, float, Sequence[Union[int, float]]],
            step_size: Union[int, float] = 0) -> float | np.ndarray:
        """
        Cumulant generating function.

        Compute the cumulant generating function K(t) = log(M(t)) = log(E[exp(tX)]).

        Parameters
        ----------
        t : int, float, or array-like
            Point(s) at which to evaluate the CGF.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero, indicates that
            input values are evenly spaced, enabling computational shortcuts.

        Returns
        -------
        float or ndarray
            CGF value(s) at t. Returns float for scalar input,
            ndarray for array-like input.

        Examples
        --------
        > dist = Normal(mu=0, sigma=1)
        > dist.cgf(0)
        0.0
        > dist.cgf([0, 0.5, 1])
        array([0., 0.125, 0.5])

        Notes
        -----
        For a normal distribution with mean mu and standard deviation sigma:

        . math::
            K(t) = \\mu t + \\frac{\\sigma^2 t^2}{2}

        For large arrays (above the CUDA threshold), computation automatically
        uses GPU acceleration if available.
        """

        validated_input = self._validate_inputs(_input=t, input_name="t")
        if isinstance(validated_input, (int, float)):
            return _core.normal_cgf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_cgf"):
            return _core.normal_cgf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_cgf_cpu(validated_input, self.mu, self.sigma, step_size)

    def sample(self) -> float:
        """
        Generate a random sample from the normal distribution.

        Returns
        -------
        float
            A single random value drawn from N(mu, sigma^2).

        Notes
        -----
        This method uses the underlying C++ core for efficient sampling.
        """

        return _core.normal_sample(self.mu, self.sigma)

    def log_sample(self) -> float:
        """
        Generate a random sample and return its natural logarithm.

        Returns
        -------
        float
            The natural logarithm of a single random value drawn from N(mu, sigma^2).

        Notes
        -----
        Useful when working with log-transformed likelihoods or log-domain calculations.
        """

        return _core.normal_log_sample(self.mu, self.sigma)

    def z_score(self, x: Union[int, float]) -> float:
        """
        Compute the z-score of a value relative to this normal distribution.

        Parameters
        ----------
        x : int or float
            The value(s) for which to compute the z-score.

        Returns
        -------
        float
            The z-score(s) corresponding to the input value(s):
            z = (x - mu) / sigma

        Notes
        -----
        Accepts both scalars and sequences. Internally validates inputs and
        delegates the computation to the optimized C++ core.
        """

        validated_input = self._validate_inputs(_input=x, input_name="x")
        return _core.z_score(validated_input, self.mu, self.sigma)

    # --------------
    # Scalar Static Methods
    # --------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float, Sequence[Union[int, float]]], mu: Union[int, float],
                    sigma: Union[int, float]) -> float:
        """
        Compute the probability density function (PDF) at a scalar value.

        Parameters
        ----------
        x : int or float
            Point at which to evaluate the PDF.
        mu : int or float
            Mean of the normal distribution.
        sigma : int or float
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            PDF evaluated at `x`.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.normal_pdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def _logpdf_scalar(cls, x: Union[int, float, Sequence[Union[int, float]]], mu: Union[int, float],
                       sigma: Union[int, float]) -> float:
        """
        Compute the natural logarithm of the PDF at a scalar value.

        Parameters
        ----------
        x : int or float
            Point at which to evaluate the log-PDF.
        mu : int or float
            Mean of the normal distribution.
        sigma : int or float
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            logpdf evaluated at `x`.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.normal_logpdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float, Sequence[Union[int, float]]], mu: Union[int, float],
                    sigma: Union[int, float]) -> float:
        """
        Compute the cumulative distribution function (CDF) at a scalar value.

        Parameters
        ----------
        x : int or float
            Point at which to evaluate the CDF.
        mu : int or float
            Mean of the normal distribution.
        sigma : int or float
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            CDF evaluated at `x`.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.normal_cdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def _mean(cls, mu: Union[int, float, Sequence[Union[int, float]]]) -> float:
        """
        Return the mean of the normal distribution.

        Parameters
        ----------
        mu : int or float
            Mean of the distribution.

        Returns
        -------
        float
            Mean of the distribution.
        """

        cls._validate_params(mu=mu)
        return _core.normal_mean(float(mu))

    @classmethod
    def _variance(cls, sigma: Union[int, float, Sequence[Union[int, float]]]) -> float:
        """
        Return the variance of the normal distribution.

        Parameters
        ----------
        sigma : int or float
            Standard deviation of the distribution.

        Returns
        -------
        float
            Variance (sigma squared) of the distribution.
        """

        cls._validate_params(sigma=sigma)
        return _core.normal_variance(float(sigma))

    @classmethod
    def _stddev(cls, sigma: Union[int, float, Sequence[Union[int, float]]]) -> float:
        """
        Return the standard deviation of the normal distribution.

        Parameters
        ----------
        sigma : int or float
            Standard deviation of the distribution.

        Returns
        -------
        float
            Standard deviation of the distribution.
        """

        cls._validate_params(sigma=sigma)
        return _core.normal_stddev(float(sigma))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float, Sequence[Union[int, float]]], mu: Union[int, float],
                    sigma: Union[int, float]) -> float:
        """
        Compute the moment-generating function (MGF) at a scalar value.

        Parameters
        ----------
        t : int or float
            Point at which to evaluate the MGF.
        mu : int or float
            Mean of the normal distribution.
        sigma : int or float
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            MGF evaluated at `t`.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.normal_mgf_scalar(float(t), float(mu), float(sigma))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float, Sequence[Union[int, float]]], mu: Union[int, float],
                    sigma: Union[int, float]) -> float:
        """
        Compute the cumulant-generating function (CGF) at a scalar value.

        Parameters
        ----------
        t : int or float
            Point at which to evaluate the CGF.
        mu : int or float
            Mean of the normal distribution.
        sigma : int or float
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            CGF evaluated at `t`.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.normal_cgf_scalar(float(t), float(mu), float(sigma))

    @classmethod
    def _sample(cls, mu: Union[int, float], sigma: Union[int, float]) -> float:
        """
        Draw a single random sample from the normal distribution.

        Parameters
        ----------
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution.

        Returns
        -------
        float
            Random sample.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_sample(float(mu), float(sigma))

    @classmethod
    def _log_sample(cls, mu: Union[int, float], sigma: Union[int, float]) -> float:
        """
        Draw a single sample and return its natural logarithm.

        Parameters
        ----------
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution.

        Returns
        -------
        float
            Logarithm of a random sample.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_log_sample(float(mu), float(sigma))

    @classmethod
    def _z_score(cls, x: Union[int, float, Sequence[Union[int, float]]], mu: Union[int, float],
                 sigma: Union[int, float]) -> float:
        """
        Compute the z-score of a value relative to the distribution.

        Parameters
        ----------
        x : int or float
            Value to compute the z-score for.
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution.

        Returns
        -------
        float
            Z-score of `x`.
        """

        cls._validate_inputs(_input=x, input_name="x")
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.z_score(float(x), float(mu), float(sigma))

    # --------------------
    # CPU Static Methods
    # --------------------
    @classmethod
    def _pdf_cpu(cls, x: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                 step_size: Union[int, float] = 0) -> NDArray[np.float64]:
        """
        Probability density function (CPU vectorized).

        Compute PDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF.
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution. Must be positive.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and x values are
            evenly spaced, enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            PDF values at each point in x.

        Examples
        --------
        > Normal._pdf_cpu([0, 1, 2], mu=0, sigma=1)
        array([0.39894228, 0.24197072, 0.05399097])
        > # With step_size for evenly-spaced grid
        > Normal._pdf_cpu(np.linspace(-3, 3, 100), mu=0, sigma=1, step_size=0.0606)
        array([...])

        Notes
        -----
        This method is automatically selected for arrays below the CUDA threshold.
        Direct use is typically not necessary unless you want to force CPU computation.
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_pdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def _logpdf_cpu(cls, x: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                    step_size: Union[int, float] = 0) -> NDArray[np.float64]:
        """
        Log probability density function (CPU vectorized).

        Compute log PDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the log PDF.
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution. Must be positive.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and x values are
            evenly spaced, enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            Log PDF values at each point in x.

        Examples
        --------
        > Normal._logpdf_cpu([0, 1, 2], mu=0, sigma=1)
        array([-0.91893853, -1.41893853, -2.91893853])
        > # With step_size for evenly-spaced grid
        > Normal._logpdf_cpu(np.linspace(-3, 3, 100), mu=0, sigma=1, step_size=0.0606)
        array([...])
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_logpdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def _cdf_cpu(cls, x: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                 step_size: Union[int, float] = 0) -> NDArray[np.float64]:
        """
        Cumulative distribution function (CPU vectorized).

        Compute CDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the CDF.
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution. Must be positive.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and x values are
            evenly spaced, enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            CDF values at each point in x, in range [0, 1].

        Examples
        --------
        > Normal._cdf_cpu([−1, 0, 1], mu=0, sigma=1)
        array([0.15865525, 0.5, 0.84134475])
        > # With step_size for evenly-spaced grid
        > Normal._cdf_cpu(np.linspace(-3, 3, 100), mu=0, sigma=1, step_size=0.0606)
        array([...])
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_cdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def _mgf_cpu(cls, t: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                 step_size: Union[int, float] = 0) -> NDArray[np.float64]:
        """
        Moment generating function (CPU vectorized).

        Compute MGF for an array of values using CPU optimization.

        Parameters
        ----------
        t : array-like
            Points at which to evaluate the MGF.
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution. Must be positive.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and t values are
            evenly spaced, enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            MGF values at each point in t.

        Examples
        --------
        > Normal._mgf_cpu([0, 0.5, 1], mu=0, sigma=1)
        array([1., 1.13315, 1.64872])
        > # With step_size for evenly-spaced grid
        > Normal._mgf_cpu(np.linspace(0, 2, 100), mu=0, sigma=1, step_size=0.0202)
        array([...])
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_mgf_cpu(t, mu, sigma, step_size)

    @classmethod
    def _cgf_cpu(cls, t: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                 step_size: Union[int, float] = 0) -> NDArray[np.float64]:
        """
        Cumulant generating function (CPU vectorized).

        Compute CGF for an array of values using CPU optimization.

        Parameters
        ----------
        t : array-like
            Points at which to evaluate the CGF.
        mu : int or float
            Mean of the distribution.
        sigma : int or float
            Standard deviation of the distribution. Must be positive.
        step_size : int or float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and t values are
            evenly spaced, enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            CGF values at each point in t.

        Examples
        --------
        > Normal._cgf_cpu([0, 0.5, 1], mu=0, sigma=1)
        array([0., 0.125, 0.5])
        > # With step_size for evenly-spaced grid
        > Normal._cgf_cpu(np.linspace(0, 2, 100), mu=0, sigma=1, step_size=0.0202)
        array([...])
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_cgf_cpu(t, mu, sigma, step_size)

    # -------------------
    # CUDA Static Methods
    # -------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def _pdf_cuda(cls, x: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                      step_size: Union[int, float] = 0) -> NDArray[np.float64]:
            """
            Probability density function (CUDA accelerated).

            Compute PDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the PDF.
            mu : int or float
                Mean of the distribution.
            sigma : int or float
                Standard deviation of the distribution. Must be positive.
            step_size : int or float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and x values are
                evenly spaced, enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                PDF values at each point in x.

            Examples
            --------
            > Normal._pdf_cuda(np.arange(1000000), mu=0, sigma=1)
            array([...])  # Fast GPU computation

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA
            threshold. Requires CUDA-capable hardware. If CUDA is not available,
            calling this method directly will raise a RuntimeError.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_pdf_cuda(x, mu, sigma, step_size)

        @classmethod
        def _logpdf_cuda(cls, x: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                         step_size: Union[int, float] = 0) -> NDArray[np.float64]:
            """
            Log probability density function (CUDA accelerated).

            Compute log PDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the log PDF.
            mu : int or float
                Mean of the distribution.
            sigma : int or float
                Standard deviation of the distribution. Must be positive.
            step_size : int or float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and x values are
                evenly spaced, enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                Log PDF values at each point in x.

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA
            threshold. Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_logpdf_cuda(x, mu, sigma, step_size)

        @classmethod
        def _cdf_cuda(cls, x: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                      step_size: Union[int, float] = 0) -> NDArray[np.float64]:
            """
            Cumulative distribution function (CUDA accelerated).

            Compute CDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the CDF.
            mu : int or float
                Mean of the distribution.
            sigma : int or float
                Standard deviation of the distribution. Must be positive.
            step_size : int or float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and x values are
                evenly spaced, enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                CDF values at each point in x, in range [0, 1].

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA
            threshold. Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_cdf_cuda(x, mu, sigma, step_size)

        @classmethod
        def _mgf_cuda(cls, t: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                      step_size: Union[int, float] = 0) -> NDArray[np.float64]:
            """
            Moment generating function (CUDA accelerated).

            Compute MGF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            t : array-like
                Points at which to evaluate the MGF.
            mu : int or float
                Mean of the distribution.
            sigma : int or float
                Standard deviation of the distribution. Must be positive.
            step_size : int or float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and t values are
                evenly spaced, enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                MGF values at each point in t.

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA
            threshold. Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_mgf_cuda(t, mu, sigma, step_size)

        @classmethod
        def _cgf_cuda(cls, t: Sequence[float], mu: Union[int, float], sigma: Union[int, float],
                      step_size: Union[int, float] = 0) -> NDArray[np.float64]:
            """
            Cumulant generating function (CUDA accelerated).
            
            Compute CGF for an array of values using GPU/CUDA acceleration.
            
            Parameters
            ----------
            t : array-like
                Points at which to evaluate the CGF.
            mu : int or float
                Mean of the distribution.
            sigma : int or float
                Standard deviation of the distribution. Must be positive.
            step_size : int or float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and t values are
                evenly spaced, enables computational shortcuts for better performance.
            
            Returns
            -------
            ndarray
                CGF values at each point in t.
            
            Notes
            -----
            This method is automatically selected for large arrays above the CUDA 
            threshold. Requires CUDA-capable hardware.
            
            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_cgf_cuda(t, mu, sigma, step_size)
    else:
        @classmethod
        def _pdf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def _logpdf_cuda(cls, *args, **kwargs):
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
