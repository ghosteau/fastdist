# python/distributions/uniform.py
try:
    from fastdist import _fastdist as _core
    from fastdist import config
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

from . import Real, Sequence, Union, np, NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'uniform_pdf_cuda')


class Uniform:
    """
    Continuous Uniform Distribution.

    Represents a uniform distribution over the interval [a, b], where all values
    within the interval are equally likely.

    Parameters
    ----------
    a : Real
        Lower bound of the distribution. Must be less than `b`.
    b : Real
        Upper bound of the distribution. Must be greater than `a`.

    Notes
    -----
    Uses a C++ backend for high-performance computation if available. Supports
    both scalar and array inputs for PDF, CDF, MGF, and CGF computations.

    Example
    -------
    >>> dist = Uniform(a=0, b=1)
    >>> dist.a
    0.0
    >>> dist.b
    1.0
    >>> dist.pdf(0.5)
    1.0
    """

    # Magic Methods
    __slots__ = ("_a", "_b")

    def __init__(self, a: Real, b: Real):
        self._validate_params(a=a, b=b)
        self._a = float(a)
        self._b = float(b)

    @property
    def a(self):
        """
        float: The lower bound of the uniform distribution.

        Notes
        -----
        Can be read or updated via the setter. Updating the value will re-validate
        that `a` is less than the current `b`.

        Example
        -------
        >>> dist = Uniform(a=0, b=1)
        >>> dist.a
        0.0
        """

        return self._a

    @property
    def b(self):
        """
        float: The upper bound of the uniform distribution.

        Notes
        -----
        Can be read or updated via the setter. Updating the value will re-validate
        that `b` is greater than the current `a`.

        Example
        -------
        >>> dist = Uniform(a=0, b=1)
        >>> dist.b
        1.0
        """

        return self._b

    @a.setter
    def a(self, value):
        """
        Set the lower bound of the distribution.

        Parameters
        ----------
        value : Real
            New lower bound; must be less than the current upper bound `b`.

        Raises
        ------
        TypeError
            If `value` is not a real number.
        ValueError
            If `value` is greater than or equal to the current `b`.

        Example
        -------
        >>> dist = Uniform(a=0, b=1)
        >>> dist.a = 0.2
        >>> dist.a
        0.2
        """

        self._validate_params(a=value)
        self._a = float(value)

    @b.setter
    def b(self, value):
        """
        Set the upper bound of the distribution.

        Parameters
        ----------
        value : Real
            New upper bound; must be greater than the current lower bound `a`.

        Raises
        ------
        TypeError
            If `value` is not a real number.
        ValueError
            If `value` is less than or equal to the current `a`.

        Example
        -------
        >>> dist = Uniform(a=0, b=1)
        >>> dist.b = 2.0
        >>> dist.b
        2.0
        """

        self._validate_params(b=value)
        self._b = float(value)

    def __repr__(self):
        return f"Uniform(a={self.a}, b={self.b})"

    @staticmethod
    def _validate_params(a: Union[Real, None] = None, b: Union[Real, None] = None):
        """
        Validate parameters for a uniform distribution.

        Checks that `a` and `b` are real numbers and that `a` is less than `b`.
        This method is used internally by the Uniform class to ensure valid distribution parameters.

        Parameters
        ----------
        a : Real or None, optional
            The lower bound of the distribution. If None, no validation is performed on `a`.
        b : Real or None, optional
            The upper bound of the distribution. If None, no validation is performed on `b`.

        Returns
        -------
        None

        Notes
        -----
        Raises exceptions if validation fails. Does not return a value.

        Raises
        ------
        TypeError
            If `a` or `b` is not a real number.
        ValueError
            If both `a` and `b` are provided and `a >= b`.
        """

        if a is not None and not isinstance(a, Real):
            raise TypeError("a must be a real number")
        if b is not None and not isinstance(b, Real):
            raise TypeError("b must be a real number")
        if a is not None and b is not None and a >= b:
            raise ValueError("a must be less than b")

    @staticmethod
    def _validate_inputs(_input: Union[Real, Sequence[Real]], input_name: str, step_size: Union[Real, None] = None) -> \
            Union[Real, np.ndarray]:
        """
        Validate input values for Uniform distribution computations.

        Ensures that the input is either a real number or a 1-dimensional numeric array.
        Optionally validates that `step_size` is a real number.

        Parameters
        ----------
        _input : Real or Sequence[Real]
            Input value(s) to validate, e.g., x-values or t-values for distribution methods.
        input_name : str
            Name of the input variable (used for error messages).
        step_size : Real or None, optional
            Optional step size used in vectorized computations; must be a real number if provided.

        Returns
        -------
        Real or np.ndarray
            Returns the validated input as a float if scalar, or as a 1-dimensional numpy array if sequenced.

        Raises
        ------
        TypeError
            If `_input` is None, or if `_input` is not a real number or numeric sequence,
            or if `step_size` is not a real number.
        ValueError
            If `_input` is a sequence but not 1-dimensional.
        """

        if _input is None:
            raise TypeError(f"{input_name} must not be None")

        if isinstance(_input, Real):
            validated = _input
        else:
            validated = Uniform._validate_array(arr=_input, input_name=input_name)
        if step_size is not None and not isinstance(step_size, Real):
            raise TypeError("step_size must be a real number")

        return validated

    @staticmethod
    def _validate_array(arr: Sequence[Real], input_name: str) -> np.ndarray:
        """
        Validate that a sequence is numeric and 1-dimensional.

        Converts the input sequence into a numpy array of type float64 and ensures
        it is one-dimensional.

        Parameters
        ----------
        arr : Sequence[Real]
            Sequence of numeric values to validate.
        input_name : str
            Name of the input variable (used in error messages).

        Returns
        -------
        np.ndarray
            A validated 1-dimensional numpy array of type float64.

        Raises
        ------
        TypeError
            If the array contains non-numeric elements.
        ValueError
            If the array is not 1-dimensional.
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
    def pdf(self, x: Union[Real, Sequence[Real]], step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Probability density function (PDF) of the uniform distribution.

        Evaluates the PDF at a scalar or array of points `x`. Uses CUDA acceleration
        if available and the input array is large.

        Parameters
        ----------
        x : Real or Sequence[Real]
            Value(s) at which to evaluate the PDF.
        step_size : Real, optional
            Step size for vectorized computations. Defaults to 0.

        Returns
        -------
        float or np.ndarray
            The PDF value(s) corresponding to `x`.

        Notes
        -----
        - For scalar input, returns a float.
        - For sequence input, returns a numpy array.
        - Automatically chooses CPU or CUDA computation based on availability
          and array size.

        Example
        -------
        >>> dist = Uniform(0, 1)
        >>> dist.pdf(0.5)
        1.0
        >>> dist.pdf([0.2, 0.5, 0.8])
        array([1., 1., 1.])
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.uniform_pdf_scalar(validated_input, self.a, self.b)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("uniform_pdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_pdf_cuda(validated_input, self.a, self.b, step_size)
        else:
            return _core.uniform_pdf_cpu(validated_input, self.a, self.b, step_size)

    def cdf(self, x: Union[Real, Sequence[Real]], step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Cumulative distribution function (CDF) of the uniform distribution.

        Computes the probability that a random variable is less than or equal to `x`.
        Supports scalar and array inputs, with optional CUDA acceleration.

        Parameters
        ----------
        x : Real or Sequence[Real]
            Value(s) at which to evaluate the CDF.
        step_size : Real, optional
            Step size for vectorized computations. Defaults to 0.

        Returns
        -------
        float or np.ndarray
            The CDF value(s) corresponding to `x`.

        Example
        -------
        >>> dist = Uniform(0, 1)
        >>> dist.cdf(0.5)
        0.5
        >>> dist.cdf([0.25, 0.5, 0.75])
        array([0.25, 0.5 , 0.75])
        """

        validated_input = self._validate_inputs(_input=x, input_name="x", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.uniform_cdf_scalar(validated_input, self.a, self.b)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("uniform_cdf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_cdf_cuda(validated_input, self.a, self.b, step_size)
        else:
            return _core.uniform_cdf_cpu(validated_input, self.a, self.b, step_size)

    def mean(self, a: Union[Real, None] = None, b: Union[Real, None] = None) -> Real:
        """
        Mean (expected value) of the uniform distribution.

        Parameters
        ----------
        a : Real, optional
            Lower bound to compute mean. Defaults to the instance's `a`.
        b : Real, optional
            Upper bound to compute mean. Defaults to the instance's `b`.

        Returns
        -------
        float
            The mean of the uniform distribution: (a + b) / 2.

        Example
        -------
        >>> dist = Uniform(0, 2)
        >>> dist.mean()
        1.0
        """

        if a is None or b is None:
            a = self.a
            b = self.b
        else:
            self._validate_params(a=a, b=b)
        return _core.uniform_mean(a, b)

    def variance(self, a: Union[Real, None] = None, b: Union[Real, None] = None) -> Real:
        """
        Variance of the uniform distribution.

        Parameters
        ----------
        a : Real, optional
            Lower bound to compute variance. Defaults to the instance's `a`.
        b : Real, optional
            Upper bound to compute variance. Defaults to the instance's `b`.

        Returns
        -------
        float
            The variance: (b - a)^2 / 12.

        Example
        -------
        >>> dist = Uniform(0, 2)
        >>> dist.variance()
        0.3333333333333333
        """

        if a is None or b is None:
            a = self.a
            b = self.b
        else:
            self._validate_params(a=a, b=b)
        return _core.uniform_variance(a, b)

    def stddev(self, a: Union[Real, None] = None, b: Union[Real, None] = None) -> Real:
        """
        Standard deviation of the uniform distribution.

        Parameters
        ----------
        a : Real, optional
            Lower bound to compute stddev. Defaults to the instance's `a`.
        b : Real, optional
            Upper bound to compute stddev. Defaults to the instance's `b`.

        Returns
        -------
        float
            Standard deviation: sqrt(variance).

        Example
        -------
        >>> dist = Uniform(0, 2)
        >>> dist.stddev()
        0.5773502691896257
        """

        if a is None or b is None:
            a = self.a
            b = self.b
        else:
            self._validate_params(a=a, b=b)
        return _core.uniform_stddev(a, b)

    def mgf(self, t: Union[Real, Sequence[Real]], step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Moment-generating function (MGF) of the uniform distribution.

        Evaluates MGF at a scalar or array of values `t`. Supports CUDA acceleration
        for large arrays.

        Parameters
        ----------
        t : Real or Sequence[Real]
            Value(s) at which to evaluate the MGF.
        step_size : Real, optional
            Step size for vectorized computations. Defaults to 0.

        Returns
        -------
        float or np.ndarray
            MGF value(s) corresponding to `t`.

        Example
        -------
        >>> dist = Uniform(0, 1)
        >>> dist.mgf(0.5)
        1.297442541400256
        >>> dist.mgf([0, 0.5, 1])
        array([1, 1.29744254, 1.71828183])
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.uniform_mgf_scalar(validated_input, self.a, self.b)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("uniform_mgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_mgf_cuda(validated_input, self.a, self.b, step_size)
        else:
            return _core.uniform_mgf_cpu(validated_input, self.a, self.b, step_size)

    def cgf(self, t: Union[Real, Sequence[Real]], step_size: Real = 0) -> Union[float, np.ndarray]:
        """
        Cumulant-generating function (CGF) of the uniform distribution.

        Computes the logarithm of the MGF at a scalar or array of `t` values.
        Uses CUDA acceleration if available.

        Parameters
        ----------
        t : Real or Sequence[Real]
            Value(s) at which to evaluate the CGF.
        step_size : Real, optional
            Step size for vectorized computations. Defaults to 0.

        Returns
        -------
        float or np.ndarray
            CGF value(s) corresponding to `t`.

        Example
        -------
        >>> dist = Uniform(0, 1)
        >>> dist.cgf(0.5)
        0.2600947485
        >>> dist.cgf([0, 0.5, 1])
        array([0, 0.26009475, 0.54132485])
        """

        validated_input = self._validate_inputs(_input=t, input_name="t", step_size=step_size)
        if isinstance(validated_input, Real):
            return _core.uniform_cgf_scalar(validated_input, self.a, self.b)
        elif _CUDA_AVAILABLE and len(validated_input) > config.get_cuda_threshold("uniform_cgf"):
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_cgf_cuda(validated_input, self.a, self.b, step_size)
        else:
            return _core.uniform_cgf_cpu(validated_input, self.a, self.b, step_size)

    def sample(self) -> Real:
        """
        Draw a random sample from the uniform distribution.

        Returns
        -------
        float
            A single random value drawn uniformly from [a, b].

        Example
        -------
        >>> dist = Uniform(0, 1)
        >>> dist.sample()
        0.7324  # Example output; random
        """

        return _core.uniform_sample(self.a, self.b)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Real, a: Real, b: Real) -> Real:
        """
        Compute the PDF of the uniform distribution at a single scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the PDF.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.

        Returns
        -------
        float
            PDF value at `x`.

        Raises
        ------
        TypeError
            If `x`, `a`, or `b` are not real numbers.
        ValueError
            If `a >= b`.

        Example
        -------
        >>> Uniform._pdf_scalar(0.5, 0, 1)
        1.0
        """

        cls._validate_params(a=a, b=b)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.uniform_pdf_scalar(float(x), float(a), float(b))

    @classmethod
    def _cdf_scalar(cls, x: Real, a: Real, b: Real) -> Real:
        """
        Compute the CDF of the uniform distribution at a single scalar value.

        Parameters
        ----------
        x : Real
            Point at which to evaluate the CDF.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.

        Returns
        -------
        float
            CDF value at `x`.

        Example
        -------
        >>> Uniform._cdf_scalar(0.5, 0, 1)
        0.5
        """

        cls._validate_params(a=a, b=b)
        cls._validate_inputs(_input=x, input_name="x")
        return _core.uniform_cdf_scalar(float(x), float(a), float(b))

    @classmethod
    def _mgf_scalar(cls, t: Real, a: Real, b: Real) -> Real:
        """
        Compute the moment-generating function (MGF) at a single scalar value.

        Parameters
        ----------
        t : Real
            Point at which to evaluate the MGF.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.

        Returns
        -------
        float
            MGF value at `t`.

        Example
        -------
        >>> Uniform._mgf_scalar(0.5, 0, 1)
        1.297442541400256
        """

        cls._validate_params(a=a, b=b)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.uniform_mgf_scalar(float(t), float(a), float(b))

    @classmethod
    def _cgf_scalar(cls, t: Real, a: Real, b: Real) -> Real:
        """
        Compute the cumulant-generating function (CGF) at a single scalar value.

        Parameters
        ----------
        t : Real
            Point at which to evaluate the CGF.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.

        Returns
        -------
        float
            CGF value at `t`.

        Example
        -------
        >>> Uniform._cgf_scalar(0.5, 0, 1)
        0.2600947485
        """

        cls._validate_params(a=a, b=b)
        cls._validate_inputs(_input=t, input_name="t")
        return _core.uniform_cgf_scalar(float(t), float(a), float(b))

    # ------------------------------------------------------------------------------------------------------------------
    # Batch Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_cpu(cls, x: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
        """
        Compute the PDF for a sequence of values using CPU computation.

        Parameters
        ----------
        x : Sequence[Real]
            Array of values to evaluate.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.
        step_size : Real, optional
            Step size for computation (default 0.0).

        Returns
        -------
        np.ndarray
            PDF values for each element in `x`.

        Example
        -------
        >>> Uniform._pdf_cpu([0, 0.5, 1], 0, 1)
        array([1., 1., 1.])
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(a=a, b=b)
        return _core.uniform_pdf_cpu(x, a, b, step_size)

    @classmethod
    def _cdf_cpu(cls, x: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
        """
        Compute the CDF for a sequence of values using CPU computation.

        Parameters
        ----------
        x : Sequence[Real]
            Array of values to evaluate.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.
        step_size : Real, optional
            Step size for computation (default 0.0).

        Returns
        -------
        np.ndarray
            CDF values for each element in `x`.

        Example
        -------
        >>> Uniform._cdf_cpu([0, 0.5, 1], 0, 1)
        array([0., 0.5, 1.])
        """

        cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
        cls._validate_params(a=a, b=b)
        return _core.uniform_cdf_cpu(x, a, b, step_size)

    @classmethod
    def _mgf_cpu(cls, t: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
        """
        Compute the MGF for a sequence of values using CPU computation.

        Parameters
        ----------
        t : Sequence[Real]
            Array of points at which to evaluate the MGF.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.
        step_size : Real, optional
            Step size for computation (default 0.0).

        Returns
        -------
        np.ndarray
            MGF values for each element in `t`.

        Example
        -------
        >>> Uniform._mgf_cpu([0, 0.5, 1], 0, 1)
        array([1., 1.29744254, 1.71828183])
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(a=a, b=b)
        return _core.uniform_mgf_cpu(t, a, b, step_size)

    @classmethod
    def _cgf_cpu(cls, t: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
        """
        Compute the CGF for a sequence of values using CPU computation.

        Parameters
        ----------
        t : Sequence[Real]
            Array of points at which to evaluate the CGF.
        a : Real
            Lower bound of the distribution.
        b : Real
            Upper bound of the distribution.
        step_size : Real, optional
            Step size for computation (default 0.0).

        Returns
        -------
        np.ndarray
            CGF values for each element in `t`.

        Example
        -------
        >>> Uniform._cgf_cpu([0, 0.5, 1], 0, 1)
        array([0., 0.26009475, 0.54132485])
        """

        cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
        cls._validate_params(a=a, b=b)
        return _core.uniform_cgf_cpu(t, a, b, step_size)

    # ------------------------------------------------------------------------------------------------------------------
    # CUDA Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def _pdf_cuda(cls, x: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
            """
            Compute the PDF for a sequence of values using CUDA acceleration.

            Parameters
            ----------
            x : Sequence[Real]
                Array of values to evaluate.
            a : Real
                Lower bound of the distribution.
            b : Real
                Upper bound of the distribution.
            step_size : Real, optional
                Step size for computation (default 0.0).

            Returns
            -------
            np.ndarray
                PDF values for each element in `x`.

            Notes
            -----
            Requires CUDA support and the `_fastdist` backend.

            Example
            -------
            >>> Uniform._pdf_cuda([0, 0.5, 1], 0, 1)
            array([1., 1., 1.])
            """

            cls._validate_params(a=a, b=b)
            validated_input = cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_pdf_cuda(validated_input, a, b, step_size)

        @classmethod
        def _cdf_cuda(cls, x: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
            """
            Compute the CDF for a sequence of values using CUDA acceleration.

            Parameters
            ----------
            x : Sequence[Real]
                Array of values to evaluate.
            a : Real
                Lower bound of the distribution.
            b : Real
                Upper bound of the distribution.
            step_size : Real, optional
                Step size for computation (default 0.0).

            Returns
            -------
            np.ndarray
                CDF values for each element in `x`.

            Example
            -------
            >>> Uniform._cdf_cuda([0, 0.5, 1], 0, 1)
            array([0., 0.5, 1.])
            """

            cls._validate_params(a=a, b=b)
            validated_input = cls._validate_inputs(_input=x, input_name="x", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_cdf_cuda(validated_input, a, b, step_size)

        @classmethod
        def _mgf_cuda(cls, t: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
            """
            Compute the MGF for a sequence of values using CUDA acceleration.

            Parameters
            ----------
            t : Sequence[Real]
                Array of points at which to evaluate the MGF.
            a : Real
                Lower bound of the distribution.
            b : Real
                Upper bound of the distribution.
            step_size : Real, optional
                Step size for computation (default 0.0).

            Returns
            -------
            np.ndarray
                MGF values for each element in `t`.

            Example
            -------
            >>> Uniform._mgf_cuda([0, 0.5, 1], 0, 1)
            array([1., 1.29744254, 1.71828183])
            """

            cls._validate_params(a=a, b=b)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_mgf_cuda(validated_input, a, b, step_size)

        @classmethod
        def _cgf_cuda(cls, t: Sequence[Real], a: Real, b: Real, step_size: Real = 0.0) -> NDArray[np.float64]:
            """
            Compute the CGF for a sequence of values using CUDA acceleration.

            Parameters
            ----------
            t : Sequence[Real]
                Array of points at which to evaluate the CGF.
            a : Real
                Lower bound of the distribution.
            b : Real
                Upper bound of the distribution.
            step_size : Real, optional
                Step size for computation (default 0.0).

            Returns
            -------
            np.ndarray
                CGF values for each element in `t`.

            Example
            -------
            >>> Uniform._cgf_cuda([0, 0.5, 1], 0, 1)
            array([0., 0.26009475, 0.54132485])
            """

            cls._validate_params(a=a, b=b)
            validated_input = cls._validate_inputs(_input=t, input_name="t", step_size=step_size)
            config.validate_gpu_capacity(validated_input.size, 8)

            return _core.uniform_cgf_cuda(validated_input, a, b, step_size)
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
