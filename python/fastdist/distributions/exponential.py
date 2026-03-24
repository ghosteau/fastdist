# python/distributions/exponential.py
try:
    from fastdist import _fastdist as _core
    from fastdist import config
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

from . import Real, Sequence, Union, NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'exponential_pdf_cuda')


class Exponential:
    # Magic Methods
    __slots__ = ("_lambda_",)

    def __init__(self, lambda_: Real):
        """
        Initialize an Exponential distribution.

        Parameters
        ----------
        lambda_ : Real
            Rate parameter of the distribution (must be positive).

        Raises
        ------
        TypeError
            If lambda_ is not a real number.
        ValueError
            If lambda_ is not positive.
        """

        self._validate_params(lambda_=lambda_)
        self._lambda_ = float(lambda_)

    @property
    def lambda_(self):
        """Rate parameter of the distribution."""
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value):
        self._validate_params(lambda_=value)
        self._lambda_ = float(value)

    def __repr__(self):
        return f"Exponential(lambda_={self.lambda_})"

    @staticmethod
    def _validate_params(lambda_: Real) -> None:
        """
        Validate the distribution parameters.

        Parameters
        ----------
        lambda_ : Real, optional
            Rate parameter.

        Raises
        ------
        TypeError
            If lambda_ is not a real number.
        ValueError
            If lambda_ is not positive.

        Notes
        -----
        Used internally to ensure valid parameters before performing calculations.
        """
        if not isinstance(lambda_, Real):
            raise TypeError("lambda_ must be a real number")
        if lambda_ <= 0:
            raise ValueError("lambda_ must be positive")

    @staticmethod
    def _validate_inputs(_input: Union[Real, Sequence[Real]], input_name: str,
                         step_size: Union[Real, None] = None) -> Union[Real, np.ndarray]:
        """
        Validate input values for distribution methods.

        Parameters
        ----------
        _input : Real, or sequence of int/float
            The input value(s) to validate.
        input_name : str
            Name of the input variable (used in error messages).
        step_size : Real, or sequence of int/float, optional
            Step size for vectorized operations. Must be a real number if provided.

        Returns
        -------
        Real or np.ndarray
            The validated input. Scalars are returned as-is, sequences are converted
            to a 1D NumPy array of type float64.

        Raises
        ------
        TypeError
            If `_input` or `step_size` has an invalid type.

        Notes
        -----
        This method is used internally to standardize inputs before passing them to
        scalar, CPU, or CUDA implementations.
        """

        if _input is None:
            raise TypeError(f"{input_name} must not be None")

        if isinstance(_input, Real):
            validated = float(_input)
        else:
            validated = Exponential._validate_array(arr=_input, input_name=input_name)
        if step_size is not None and not isinstance(step_size, Real):
            raise TypeError("step_size must be a real number")

        return validated

    @staticmethod
    def _validate_array(arr: Sequence[Real], input_name: str) -> np.ndarray:
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
        """
        return _CUDA_AVAILABLE

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pdf(self, x: Union[Real, Sequence[Real]],
            step_size: Real = 0) -> Union[Real, np.ndarray]:
        """
        Probability density function.

        Compute the PDF of the exponential distribution at x.

        Parameters
        ----------
        x : Real or array-like
            Point(s) at which to evaluate the PDF.
        step_size : Real, optional
            Step size for evenly spaced grids (default 0, no optimization).

        Returns
        -------
        float or np.ndarray
            PDF values at x.

        Notes
        -----
        For large arrays above the CUDA threshold, GPU acceleration is used if available.
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.exponential_pdf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("exponential_pdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_pdf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.exponential_pdf_cpu(validated_input, self.lambda_, step_size)

    def cdf(self, x: Union[Real, Sequence[Real]],
            step_size: Real = 0) -> Union[Real, np.ndarray]:
        """
        Cumulative distribution function.

        Compute the CDF of the exponential distribution at x.

        Parameters
        ----------
        x : Real or array-like
            Point(s) at which to evaluate the CDF.
        step_size : Real, optional
            Step size for evenly spaced grids (default 0, no optimization).

        Returns
        -------
        float or np.ndarray
            CDF values at x in [0, 1].

        Notes
        -----
        For large arrays above the CUDA threshold, GPU acceleration is used if available.
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.exponential_cdf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("exponential_cdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_cdf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.exponential_cdf_cpu(validated_input, self.lambda_, step_size)

    def mean(self, lambda_: Union[Real, None] = None) -> Real:
        """
        Mean (expected value) of the distribution.

        Returns
        -------
        float
            Mean of the distribution (1/lambda_).
        """

        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.exponential_mean(lambda_)

    def variance(self, lambda_: Union[Real, None] = None) -> Real:
        """
        Variance of the distribution.

        Returns
        -------
        float
            Variance of the distribution (1/lambda_^2).
        """

        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.exponential_variance(lambda_)

    def stddev(self, lambda_: Union[Real, None] = None) -> Real:
        """
        Standard deviation of the distribution.

        Returns
        -------
        float
            Standard deviation (1/lambda_).
        """

        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.exponential_stddev(lambda_)

    def mgf(self, t: Union[Real, Sequence[Real]], step_size: Real = 0) -> Union[Real, np.ndarray]:
        """
        Moment generating function.

        Compute MGF M(t) = E[exp(tX)].

        Parameters
        ----------
        t : Real or array-like
            Points at which to evaluate the MGF.
        step_size : Real, optional
            Step size for evenly spaced grids.

        Returns
        -------
        float or np.ndarray
            MGF values at t.

        Notes
        -----
        For large arrays, GPU acceleration is used if available.
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.exponential_mgf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("exponential_mgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_mgf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.exponential_mgf_cpu(validated_input, self.lambda_, step_size)

    def cgf(self, t: Union[Real, Sequence[Real]], step_size: Real = 0) -> Union[Real, np.ndarray]:
        """
        Cumulant generating function.

        Compute CGF K(t) = log(M(t)).

        Parameters
        ----------
        t : Real or array-like
            Points at which to evaluate the CGF.
        step_size : Real, optional
            Step size for evenly spaced grids.

        Returns
        -------
        float or np.ndarray
            CGF values at t.

        Notes
        -----
        For large arrays, GPU acceleration is used if available.
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.exponential_cgf_scalar(validated_input, self.lambda_)
        elif _CUDA_AVAILABLE and validated_input.size > config.get_cuda_threshold("exponential_cgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_cgf_cuda(validated_input, self.lambda_, step_size)
        else:
            return _core.exponential_cgf_cpu(validated_input, self.lambda_, step_size)

    def sample(self, lambda_: Union[Real, None] = None) -> Real:
        """
        Generate a random sample from the exponential distribution.

        Returns
        -------
        float
            Single random value drawn from Exp(lambda_).
        """

        if lambda_ is None:
            lambda_ = self.lambda_
        else:
            self._validate_params(lambda_=lambda_)
        return _core.exponential_sample(lambda_)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Real, lambda_: Real) -> Real:
        """
        Compute the probability density function (PDF) at a scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the PDF.
        lambda_ : Real
            Rate parameter of the exponential distribution.

        Returns
        -------
        float
            PDF evaluated at `x`.

        Raises
        ------
        TypeError
            If `x` or `lambda_` is not a real number.
        ValueError
            If `lambda_` is not positive.
        """

        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.exponential_pdf_scalar(float(x), float(lambda_))

    @classmethod
    def _cdf_scalar(cls, x: Real, lambda_: Real) -> Real:
        """
        Compute the cumulative distribution function (CDF) at a scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the CDF.
        lambda_ : Real
            Rate parameter of the exponential distribution.

        Returns
        -------
        float
            CDF evaluated at `x` in the range [0, 1].

        Raises
        ------
        TypeError
            If `x` or `lambda_` is not a real number.
        ValueError
            If `lambda_` is not positive.
        """

        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.exponential_cdf_scalar(float(x), float(lambda_))

    @classmethod
    def _mgf_scalar(cls, t: Real, lambda_: Real) -> Real:
        """
        Compute the moment-generating function (MGF) at a scalar value.

        Parameters
        ----------
        t : Real
            Point at which to evaluate the MGF.
        lambda_ : Real
            Rate parameter of the exponential distribution.

        Returns
        -------
        float
            MGF evaluated at `t`.

        Raises
        ------
        TypeError
            If `t` or `lambda_` is not a real number.
        ValueError
            If `lambda_` is not positive.
        """

        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.exponential_mgf_scalar(float(t), float(lambda_))

    @classmethod
    def _cgf_scalar(cls, t: Real, lambda_: Real) -> Real:
        """
        Compute the cumulant-generating function (CGF) at a scalar value.

        Parameters
        ----------
        t : Real
            Point at which to evaluate the CGF.
        lambda_ : Real
            Rate parameter of the exponential distribution.

        Returns
        -------
        float
            CGF evaluated at `t`.

        Raises
        ------
        TypeError
            If `t` or `lambda_` is not a real number.
        ValueError
            If `lambda_` is not positive.
        """

        cls._validate_params(lambda_=lambda_)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.exponential_cgf_scalar(float(t), float(lambda_))

    # ------------------------------------------------------------------------------------------------------------------
    # Batch Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_cpu(cls, x: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
        """
        Probability density function (CPU vectorized).

        Compute the PDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the PDF.
        lambda_ : float
            Rate parameter of the exponential distribution. Must be positive.
        step_size : float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and x values are evenly spaced,
            enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            PDF values at each point in x.

        Examples
        --------
        >>> Exponential._pdf_cpu([0, 1, 2], lambda_=1)
        array([1.0, 0.36787944, 0.13533528])
        >>> # With step_size for evenly-spaced grid
        >>> Exponential._pdf_cpu(np.linspace(0, 5, 100), lambda_=1, step_size=0.0505)
        array([...])

        Notes
        -----
        This method is automatically selected for arrays below the CUDA threshold.
        Direct use is typically not necessary unless you want to force CPU computation.
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_pdf_cpu(x, lambda_, step_size)

    @classmethod
    def _cdf_cpu(cls, x: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
        """
        Cumulative distribution function (CPU vectorized).

        Compute the CDF for an array of values using CPU optimization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the CDF.
        lambda_ : float
            Rate parameter of the exponential distribution. Must be positive.
        step_size : float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and x values are evenly spaced,
            enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            CDF values at each point in x, in range [0, 1].

        Examples
        --------
        >>> Exponential._cdf_cpu([0, 1, 2], lambda_=1)
        array([0.0, 0.63212056, 0.86466472])
        >>> # With step_size for evenly-spaced grid
        >>> Exponential._cdf_cpu(np.linspace(0, 5, 100), lambda_=1, step_size=0.0505)
        array([...])
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_cdf_cpu(x, lambda_, step_size)

    @classmethod
    def _mgf_cpu(cls, t: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
        """
        Moment generating function (CPU vectorized).

        Compute the MGF for an array of values using CPU optimization.

        Parameters
        ----------
        t : array-like
            Points at which to evaluate the MGF.
        lambda_ : float
            Rate parameter of the exponential distribution. Must be positive.
        step_size : float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and t values are evenly spaced,
            enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            MGF values at each point in t.

        Examples
        --------
        >>> Exponential._mgf_cpu([0, 0.5, 1], lambda_=1)
        array([1.0, 2.0, inf])
        >>> # With step_size for evenly-spaced grid
        >>> Exponential._mgf_cpu(np.linspace(0, 0.9, 100), lambda_=1, step_size=0.009)
        array([...])
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_mgf_cpu(t, lambda_, step_size)

    @classmethod
    def _cgf_cpu(cls, t: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
        """
        Cumulant generating function (CPU vectorized).

        Compute the CGF for an array of values using CPU optimization.

        Parameters
        ----------
        t : array-like
            Points at which to evaluate the CGF.
        lambda_ : float
            Rate parameter of the exponential distribution. Must be positive.
        step_size : float, optional
            Step size parameter for optimizing computation on regular grids.
            Default is 0 (no optimization). When non-zero and t values are evenly spaced,
            enables computational shortcuts for better performance.

        Returns
        -------
        ndarray
            CGF values at each point in t.

        Examples
        --------
        >>> Exponential._cgf_cpu([0, 0.5, 1], lambda_=1)
        array([0.0, 0.69314718, inf])
        >>> # With step_size for evenly-spaced grid
        >>> Exponential._cgf_cpu(np.linspace(0, 0.9, 100), lambda_=1, step_size=0.009)
        array([...])
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(lambda_=lambda_)
        return _core.exponential_cgf_cpu(t, lambda_, step_size)

    # ------------------------------------------------------------------------------------------------------------------
    # CUDA Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def _pdf_cuda(cls, x: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
            """
            Probability density function (CUDA accelerated).

            Compute PDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the PDF.
            lambda_ : float
                Rate parameter of the exponential distribution. Must be positive.
            step_size : float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and x values are evenly spaced,
                enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                PDF values at each point in x.

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA threshold.
            Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_pdf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _cdf_cuda(cls, x: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
            """
            Cumulative distribution function (CUDA accelerated).

            Compute CDF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            x : array-like
                Points at which to evaluate the CDF.
            lambda_ : float
                Rate parameter of the exponential distribution. Must be positive.
            step_size : float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and x values are evenly spaced,
                enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                CDF values at each point in x, in range [0, 1].

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA threshold.
            Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_cdf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _mgf_cuda(cls, t: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
            """
            Moment generating function (CUDA accelerated).

            Compute MGF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            t : array-like
                Points at which to evaluate the MGF.
            lambda_ : float
                Rate parameter of the exponential distribution. Must be positive.
            step_size : float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and t values are evenly spaced,
                enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                MGF values at each point in t.

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA threshold.
            Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_mgf_cuda(validated_input, lambda_, step_size)

        @classmethod
        def _cgf_cuda(cls, t: Sequence[Real], lambda_: Real, step_size: Real = 0) -> NDArray[np.float64]:
            """
            Cumulant generating function (CUDA accelerated).

            Compute CGF for an array of values using GPU/CUDA acceleration.

            Parameters
            ----------
            t : array-like
                Points at which to evaluate the CGF.
            lambda_ : float
                Rate parameter of the exponential distribution. Must be positive.
            step_size : float, optional
                Step size parameter for optimizing computation on regular grids.
                Default is 0 (no optimization). When non-zero and t values are evenly spaced,
                enables computational shortcuts for better performance.

            Returns
            -------
            ndarray
                CGF values at each point in t.

            Notes
            -----
            This method is automatically selected for large arrays above the CUDA threshold.
            Requires CUDA-capable hardware.

            Raises
            ------
            RuntimeError
                If CUDA support is not available in the installed package.
            """

            cls._validate_params(lambda_=lambda_)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.exponential_cgf_cuda(validated_input, lambda_, step_size)
    else:
        @classmethod
        def _pdf_cuda(cls, *args, **kwargs):
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
