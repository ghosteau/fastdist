# python/distributions/normal.py
try:
    from fastdist import _fastdist as _core
    from fastdist import config
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import math
from numbers import Real
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'normal_pdf_cuda')


class Normal:
    """
    Normal (Gaussian) distribution class.

    Provides PDF, CDF, and other operations for the normal distribution.
    Supports both scalar and vectorized operations, with optional CUDA acceleration
    for large arrays when available.

    Parameters
    ----------
    mu : Real
        Mean (location) of the distribution. Must be finite.
    sigma : Real
        Standard deviation (scale) of the distribution. Must be positive and finite.

    Attributes
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation of the distribution.

    Notes
    -----
    This class validates input parameters and supports automatic CPU/CUDA acceleration
    for efficient computation.

    Example
    -------
    >>> dist = Normal(mu=0, sigma=1)
    >>> dist.mu
    0.0
    >>> dist.sigma
    1.0
    >>> dist._validate_params(mu=0)
    >>> Normal.is_cuda_available()
    True or False

    Raises
    ------
    ValueError
        If sigma is non-positive or any parameter is not finite.
    TypeError
        If mu or sigma is not a real number.
    """

    # Magic Methods
    __slots__ = ("_mu", "_sigma")

    def __init__(self, mu: Real, sigma: Real):
        """
        Initialize a Normal distribution with specified mean and standard deviation.

        Parameters
        ----------
        mu : Real
            Mean of the distribution.
        sigma : Real
            Standard deviation of the distribution.
        """

        self._validate_params(mu=mu, sigma=sigma)
        self._mu = float(mu)
        self._sigma = float(sigma)

    @property
    def mu(self):
        """
        Real: Mean of the distribution.
        """

        return self._mu

    @property
    def sigma(self):
        """
        Real: Standard deviation of the distribution.
        """

        return self._sigma

    @mu.setter
    def mu(self, value: Real):
        """
        Set a new mean for the distribution.

        Parameters
        ----------
        value : Real
            New mean value.

        Raises
        ------
        TypeError, ValueError
            If the value is invalid.
        """

        self._validate_params(mu=value)
        self._mu = float(value)

    @sigma.setter
    def sigma(self, value: Real):
        """
        Set a new standard deviation for the distribution.

        Parameters
        ----------
        value : Real
            New standard deviation value.

        Raises
        ------
        TypeError, ValueError
            If the value is invalid.
        """

        self._validate_params(sigma=value)
        self._sigma = float(value)

    def __repr__(self):
        """
        Return a string representation of the distribution.
        """
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    @staticmethod
    def _validate_params(mu: Union[Real, None] = None,
                         sigma: Union[Real, None] = None):
        """
        Validate the distribution parameters.

        Parameters
        ----------
        mu : Real, optional
            Mean of the distribution.
        sigma : Real, optional
            Standard deviation of the distribution.

        Raises
        ------
        TypeError
            If mu or sigma is not a real number.
        ValueError
            If mu is not finite or sigma is non-positive or non-finite.

        Notes
        -----
        Used internally to ensure valid parameters before performing calculations.
        """

        if mu is not None:
            if not isinstance(mu, Real):
                raise TypeError("mu must be a real number")
            if not math.isfinite(mu):
                raise ValueError("mu must be finite")
        if sigma is not None:
            if not isinstance(sigma, Real):
                raise TypeError("sigma must be a real number")
            if not math.isfinite(sigma):
                raise ValueError("sigma must be finite")
            if sigma <= 0:
                raise ValueError("sigma must be positive")

    @staticmethod
    def _validate_inputs(_input: Union[Real, Sequence[Real]], input_name: str,
                         step_size: Union[Real, None] = None) -> Union[Real, np.ndarray]:
        """
        Validate inputs for distribution methods.

        Parameters
        ----------
        _input : Real or sequence of Real
            Input value(s) to validate.
        input_name : str
            Name of the input variable (used in error messages).
        step_size : Real, optional
            Step size for vectorized operations.

        Returns
        -------
        Real or np.ndarray
            Validated input. Scalars returned as-is, sequences converted to 1D NumPy array.

        Raises
        ------
        TypeError
            If _input or step_size has invalid type.

        Notes
        -----
        Standardizes inputs before passing them to CPU or CUDA operations.
        """

        if _input is None:
            raise TypeError(f"{input_name} must not be None")

        if isinstance(_input, Real):
            validated = _input
        else:
            validated = Normal._validate_array(arr=_input, input_name=input_name)
        if step_size is not None and not isinstance(step_size, Real):
            raise TypeError("step_size must be a real number")

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

    # ----------------
    # Instance Methods
    # ----------------
    def pdf(self, x: Union[Real, Sequence[Real]],
            step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Probability density function (PDF).

        Compute the value of the PDF at the specified point(s).

        Parameters
        ----------
        x : Real or array-like
            Point(s) at which to evaluate the PDF.
        step_size : Real, optional
            Step size for optimizing computations on evenly spaced inputs.
            Default is 0 (no optimization).

        Returns
        -------
        float or np.ndarray
            PDF value(s). Returns float for scalar input, ndarray for array-like input.

        Notes
        -----
        Automatically uses CUDA acceleration for large arrays if available.

        Examples
        --------
        >>> dist = Normal(mu=0, sigma=1)
        >>> dist.pdf(0)
        0.3989422804014327
        >>> dist.pdf([-1, 0, 1])
        array([0.24197072, 0.39894228, 0.24197072])
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.normal_pdf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_pdf"):
            return _core.normal_pdf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_pdf_cpu(validated_input, self.mu, self.sigma, step_size)

    def logpdf(self, x: Union[Real, Sequence[Real]],
               step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Log probability density function.

        Compute the natural logarithm of the PDF at the specified point(s).

        Parameters
        ----------
        x : Real or array-like
            Point(s) at which to evaluate the log PDF.
        step_size : Real, optional
            Step size for optimizing computations on evenly spaced inputs.
            Default is 0 (no optimization).

        Returns
        -------
        float or np.ndarray
            Log PDF value(s). Returns float for scalar input, ndarray for array-like input.

        Notes
        -----
        More numerically stable than computing log(PDF) directly.
        Automatically uses CUDA acceleration for large arrays if available.

        Examples
        --------
        >>> dist = Normal(mu=0, sigma=1)
        >>> dist.logpdf(0)
        -0.9189385332046727
        >>> dist.logpdf([0, 1, 2])
        array([-0.91893853, -1.41893853, -2.91893853])
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.normal_logpdf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_logpdf"):
            return _core.normal_logpdf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_logpdf_cpu(validated_input, self.mu, self.sigma, step_size)

    def cdf(self, x: Union[Real, Sequence[Real]],
            step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function (CDF).

        Compute the probability that a random variable X is less than or equal to x.

        Parameters
        ----------
        x : Real or array-like
            Point(s) at which to evaluate the CDF.
        step_size : Real, optional
            Step size for optimizing computations on evenly spaced inputs.
            Default is 0 (no optimization).

        Returns
        -------
        float or np.ndarray
            CDF value(s) in [0, 1]. Returns float for scalar input, ndarray for array-like input.

        Notes
        -----
        Automatically uses CUDA acceleration for large arrays if available.

        Examples
        --------
        >>> dist = Normal(mu=0, sigma=1)
        >>> dist.cdf(0)
        0.5
        >>> dist.cdf([-1, 0, 1])
        array([0.15865525, 0.5, 0.84134475])
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.normal_cdf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_cdf"):
            return _core.normal_cdf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_cdf_cpu(validated_input, self.mu, self.sigma, step_size)

    def mean(self, mu: Union[Real, None] = None) -> Real:
        """
        Mean (expected value) of the distribution.

        Parameters
        ----------
        mu : Real, optional
            Overrides the distribution's mean if provided.

        Returns
        -------
        float
            The mean of the distribution.

        Notes
        -----
        Delegates computation to the underlying C++ core.
        """

        if mu is None:
            mu = self.mu
        else:
            self._validate_params(mu=mu)
        return _core.normal_mean(mu)

    def variance(self, sigma: Union[Real, None] = None) -> Real:
        """
        Variance of the distribution.

        Parameters
        ----------
        sigma : Real, optional
            Overrides the distribution's standard deviation if provided.

        Returns
        -------
        float
            Variance, equal to sigma squared.

        Notes
        -----
        Delegates computation to the underlying C++ core.
        """

        if sigma is None:
            sigma = self.sigma
        else:
            self._validate_params(sigma=sigma)
        return _core.normal_variance(sigma)

    def stddev(self, sigma: Union[Real, None] = None) -> Real:
        """
        Standard deviation of the distribution.

        Parameters
        ----------
        sigma : Real, optional
            Overrides the distribution's standard deviation if provided.

        Returns
        -------
        float
            Standard deviation (sigma).

        Notes
        -----
        Delegates computation to the underlying C++ core.
        """

        if sigma is None:
            sigma = self.sigma
        else:
            self._validate_params(sigma=sigma)
        return _core.normal_stddev(sigma)

    def mgf(self, t: Union[Real, Sequence[Real]],
            step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Moment generating function (MGF).

        Compute M(t) = E[exp(tX)].

        Parameters
        ----------
        t : Real or array-like
            Point(s) at which to evaluate the MGF.
        step_size : Real, optional
            Step size for optimizing computations on evenly spaced inputs.

        Returns
        -------
        float or np.ndarray
            MGF value(s) at t.

        Notes
        -----
        M(t) = exp(mu * t + 0.5 * sigma^2 * t^2)
        Automatically uses CUDA acceleration for large arrays if available.

        Examples
        --------
        >>> dist = Normal(mu=0, sigma=1)
        >>> dist.mgf(0)
        1.0
        >>> dist.mgf([0, 0.5, 1])
        array([1.0, 1.13315, 1.64872])
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.normal_mgf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_mgf"):
            return _core.normal_mgf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_mgf_cpu(validated_input, self.mu, self.sigma, step_size)

    def cgf(self, t: Union[Real, Sequence[Real]],
            step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Cumulant generating function (CGF).

        Compute K(t) = log(M(t)) = log(E[exp(tX)]).

        Parameters
        ----------
        t : Real or array-like
            Point(s) at which to evaluate the CGF.
        step_size : Real, optional
            Step size for optimizing computations on evenly spaced inputs.

        Returns
        -------
        float or np.ndarray
            CGF value(s) at t.

        Notes
        -----
        K(t) = mu * t + 0.5 * sigma^2 * t^2
        Automatically uses CUDA acceleration for large arrays if available.

        Examples
        --------
        >>> dist = Normal(mu=0, sigma=1)
        >>> dist.cgf(0)
        0.0
        >>> dist.cgf([0, 0.5, 1])
        array([0.0, 0.125, 0.5])
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.normal_cgf_scalar(validated_input, self.mu, self.sigma)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("normal_cgf"):
            return _core.normal_cgf_cuda(validated_input, self.mu, self.sigma, step_size)
        else:
            return _core.normal_cgf_cpu(validated_input, self.mu, self.sigma, step_size)

    def sample(self, mu: Union[Real, None] = None, sigma: Union[Real, None] = None) -> Real:
        """
        Generate a random sample from the normal distribution.

        Parameters
        ----------
        mu : Real, optional
            Mean to use for sampling. Defaults to distribution's mu.
        sigma : Real, optional
            Standard deviation to use for sampling. Defaults to distribution's sigma.

        Returns
        -------
        float
            A single random sample from N(mu, sigma^2).

        Notes
        -----
        Uses the underlying C++ core for efficient sampling.
        """

        if sigma is None or mu is None:
            sigma = self.sigma
            mu = self.mu
        else:
            self._validate_params(mu=mu, sigma=sigma)
        return _core.normal_sample(mu, sigma)

    def log_sample(self, mu: Union[Real, None] = None, sigma: Union[Real, None] = None) -> Real:
        """
        Generate a random sample and return its natural logarithm.

        Parameters
        ----------
        mu : Real, optional
            Mean to use for sampling. Defaults to distribution's mu.
        sigma : Real, optional
            Standard deviation to use for sampling. Defaults to distribution's sigma.

        Returns
        -------
        float
            Natural logarithm of a single random sample from N(mu, sigma^2).

        Notes
        -----
        Useful for log-likelihood or log-domain calculations.
        """

        if sigma is None or mu is None:
            sigma = self.sigma
            mu = self.mu
        else:
            self._validate_params(mu=mu, sigma=sigma)
        return _core.normal_log_sample(mu, sigma)

    def z_score(self, x: Real, mu: Union[Real, None] = None, sigma: Union[Real, None] = None) -> Real:
        """
        Compute the z-score of a value relative to this normal distribution.

        Parameters
        ----------
        x : Real
            The value(s) for which to compute the z-score.
        mu : Real, optional
            Mean to use for the z-score. Defaults to the distribution's `mu`.
        sigma : Real, optional
            Standard deviation to use for the z-score. Defaults to the distribution's `sigma`.

        Returns
        -------
        float
            The z-score corresponding to `x`: z = (x - mu) / sigma.

        Notes
        -----
        Accepts scalars or sequences. Delegates computation to the optimized C++ core.
        """

        if sigma is None or mu is None:
            sigma = self.sigma
            mu = self.mu
        else:
            self._validate_params(mu=mu, sigma=sigma)

        validated_input = self._validate_inputs(_input=x, input_name="x")
        return _core.z_score(validated_input, mu, sigma)

    # ---------------------
    # Scalar Static Methods
    # ---------------------
    @classmethod
    def pdf_scalar(cls, x: Real, mu: Real, sigma: Real) -> Real:
        """
        Compute the probability density function (PDF) at a scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the PDF.
        mu : Real
            Mean of the normal distribution.
        sigma : Real
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            PDF evaluated at `x`.

        Notes
        -----
        Validates all parameters and inputs before computation. Uses the C++ core.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.normal_pdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def logpdf_scalar(cls, x: Real, mu: Real, sigma: Real) -> Real:
        """
        Compute the natural logarithm of the PDF at a scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the log-PDF.
        mu : Real
            Mean of the normal distribution.
        sigma : Real
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            log-PDF evaluated at `x`.

        Notes
        -----
        More numerically stable than computing log(PDF) directly.
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.normal_logpdf_scalar(float(x), float(mu), float(sigma))

    @classmethod
    def cdf_scalar(cls, x: Real, mu: Real, sigma: Real) -> Real:
        """
        Compute the cumulative distribution function (CDF) at a scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the CDF.
        mu : Real
            Mean of the normal distribution.
        sigma : Real
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
    def mgf_scalar(cls, t: Real, mu: Real, sigma: Real) -> Real:
        """
        Compute the moment-generating function (MGF) at a scalar value.

        Parameters
        ----------
        t : Real
            Point at which to evaluate the MGF.
        mu : Real
            Mean of the normal distribution.
        sigma : Real
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            MGF evaluated at `t`.

        Notes
        -----
        M(t) = exp(mu * t + 0.5 * sigma^2 * t^2)
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.normal_mgf_scalar(float(t), float(mu), float(sigma))

    @classmethod
    def cgf_scalar(cls, t: Real, mu: Real, sigma: Real) -> Real:
        """
        Compute the cumulant-generating function (CGF) at a scalar value.

        Parameters
        ----------
        t : Real
            Point at which to evaluate the CGF.
        mu : Real
            Mean of the normal distribution.
        sigma : Real
            Standard deviation of the normal distribution.

        Returns
        -------
        float
            CGF evaluated at `t`.

        Notes
        -----
        K(t) = mu * t + 0.5 * sigma^2 * t^2
        """

        cls._validate_params(mu=mu, sigma=sigma)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.normal_cgf_scalar(float(t), float(mu), float(sigma))

    # --------------------
    # CPU Static Methods
    # --------------------
    @classmethod
    def pdf_cpu(cls, x: Sequence[Real], mu: Real, sigma: Real,
                step_size: Real = 0) -> NDArray[np.float64]:
        """
        Probability density function (CPU vectorized).

        Compute PDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF.
        mu : Real
            Mean of the distribution.
        sigma : Real
            Standard deviation of the distribution. Must be positive.
        step_size : Real, optional
            Step size for evenly spaced inputs. Default is 0 (no optimization).

        Returns
        -------
        ndarray
            PDF values at each point in `x`.

        Notes
        -----
        Automatically used for arrays below the CUDA threshold. Direct use is
        optional when CPU computation is desired.
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_pdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def logpdf_cpu(cls, x: Sequence[Real], mu: Real, sigma: Real,
                   step_size: Real = 0) -> NDArray[np.float64]:
        """
        Log probability density function (CPU vectorized).

        Compute log PDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the log PDF.
        mu : Real
            Mean of the distribution.
        sigma : Real
            Standard deviation of the distribution. Must be positive.
        step_size : Real, optional
            Step size for evenly spaced inputs. Default is 0.

        Returns
        -------
        ndarray
            Log PDF values at each point in `x`.
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_logpdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def cdf_cpu(cls, x: Sequence[Real], mu: Real, sigma: Real,
                step_size: Real = 0) -> NDArray[np.float64]:
        """
        Cumulative distribution function (CPU vectorized).

        Compute CDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the CDF.
        mu : Real
            Mean of the distribution.
        sigma : Real
            Standard deviation of the distribution. Must be positive.
        step_size : Real, optional
            Step size for evenly spaced inputs. Default is 0.

        Returns
        -------
        ndarray
            CDF values at each point in `x`.
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_cdf_cpu(x, mu, sigma, step_size)

    @classmethod
    def mgf_cpu(cls, t: Sequence[Real], mu: Real, sigma: Real,
                step_size: Real = 0) -> NDArray[np.float64]:
        """
        Moment generating function (CPU vectorized).

        Compute MGF for an array of values using CPU optimization.

        Parameters
        ----------
        t : array-like
            Points at which to evaluate the MGF.
        mu : Real
            Mean of the distribution.
        sigma : Real
            Standard deviation of the distribution. Must be positive.
        step_size : Real, optional
            Step size for evenly spaced inputs. Default is 0.

        Returns
        -------
        ndarray
            MGF values at each point in `t`.
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_mgf_cpu(t, mu, sigma, step_size)

    @classmethod
    def cgf_cpu(cls, t: Sequence[Real], mu: Real, sigma: Real,
                step_size: Real = 0) -> NDArray[np.float64]:
        """
        Cumulant generating function (CPU vectorized).

        Compute CGF for an array of values using CPU optimization.

        Parameters
        ----------
        t : array-like
            Points at which to evaluate the CGF.
        mu : Real
            Mean of the distribution.
        sigma : Real
            Standard deviation of the distribution. Must be positive.
        step_size : Real, optional
            Step size for evenly spaced inputs. Default is 0.

        Returns
        -------
        ndarray
            CGF values at each point in `t`.
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(mu=mu, sigma=sigma)
        return _core.normal_cgf_cpu(t, mu, sigma, step_size)

    # -------------------
    # CUDA Static Methods
    # -------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def pdf_cuda(cls, x: Sequence[Real], mu: Real, sigma: Real,
                     step_size: Real = 0) -> NDArray[np.float64]:
            """
            Probability density function (CUDA accelerated).

            Compute PDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the PDF.
            mu : Real
                Mean of the distribution.
            sigma : Real
                Standard deviation of the distribution. Must be positive.
            step_size : Real, optional
                Step size for evenly spaced inputs. Default is 0 (no optimization).

            Returns
            -------
            ndarray
                PDF values at each point in `x`.

            Notes
            -----
            Automatically selected for large arrays above the CUDA threshold. Requires
            CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available.
            """

            cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_pdf_cuda(x, mu, sigma, step_size)

        @classmethod
        def logpdf_cuda(cls, x: Sequence[Real], mu: Real, sigma: Real,
                        step_size: Real = 0) -> NDArray[np.float64]:
            """
            Log probability density function (CUDA accelerated).

            Compute log PDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the log PDF.
            mu : Real
                Mean of the distribution.
            sigma : Real
                Standard deviation of the distribution. Must be positive.
            step_size : Real, optional
                Step size for evenly spaced inputs. Default is 0.

            Returns
            -------
            ndarray
                Log PDF values at each point in `x`.

            Notes
            -----
            Automatically selected for large arrays above the CUDA threshold. Requires
            CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available.
            """

            cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_logpdf_cuda(x, mu, sigma, step_size)

        @classmethod
        def cdf_cuda(cls, x: Sequence[Real], mu: Real, sigma: Real,
                     step_size: Real = 0) -> NDArray[np.float64]:
            """
            Cumulative distribution function (CUDA accelerated).

            Compute CDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the CDF.
            mu : Real
                Mean of the distribution.
            sigma : Real
                Standard deviation of the distribution. Must be positive.
            step_size : Real, optional
                Step size for evenly spaced inputs. Default is 0.

            Returns
            -------
            ndarray
                CDF values at each point in `x`, in range [0, 1].

            Notes
            -----
            Automatically selected for large arrays above the CUDA threshold. Requires
            CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available.
            """

            cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_cdf_cuda(x, mu, sigma, step_size)

        @classmethod
        def mgf_cuda(cls, t: Sequence[Real], mu: Real, sigma: Real,
                     step_size: Real = 0) -> NDArray[np.float64]:
            """
            Moment generating function (CUDA accelerated).

            Compute MGF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            t : array-like
                Points at which to evaluate the MGF.
            mu : Real
                Mean of the distribution.
            sigma : Real
                Standard deviation of the distribution. Must be positive.
            step_size : Real, optional
                Step size for evenly spaced inputs. Default is 0.

            Returns
            -------
            ndarray
                MGF values at each point in `t`.

            Notes
            -----
            Automatically selected for large arrays above the CUDA threshold. Requires
            CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available.
            """

            cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_mgf_cuda(t, mu, sigma, step_size)

        @classmethod
        def cgf_cuda(cls, t: Sequence[Real], mu: Real, sigma: Real,
                     step_size: Real = 0) -> NDArray[np.float64]:
            """
            Cumulant generating function (CUDA accelerated).

            Compute CGF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            t : array-like
                Points at which to evaluate the CGF.
            mu : Real
                Mean of the distribution.
            sigma : Real
                Standard deviation of the distribution. Must be positive.
            step_size : Real, optional
                Step size for evenly spaced inputs. Default is 0.

            Returns
            -------
            ndarray
                CGF values at each point in `t`.

            Notes
            -----
            Automatically selected for large arrays above the CUDA threshold. Requires
            CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available.
            """

            cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            cls._validate_params(mu=mu, sigma=sigma)
            return _core.normal_cgf_cuda(t, mu, sigma, step_size)
    else:
        @classmethod
        def pdf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def logpdf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def cdf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def mgf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def cgf_cuda(cls, *args, **kwargs):
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )
