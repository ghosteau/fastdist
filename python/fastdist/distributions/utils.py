# python/fastdist/distributions/utils.py
try:
    from fastdist import _fastdist as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "fastdist's compiled extension (_fastdist) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

from fastdist import config

import numpy as np
from numbers import Real
from typing import Sequence, SupportsFloat, Union, cast
from numpy.typing import ArrayLike, NDArray

# Check CUDA availability at module load time
_CUDA_AVAILABLE = hasattr(_core, 'sigmoid_cuda')


class Utils:
    """
    Statistical and combinatorial helpers.

    A namespace of stateless classmethods for quantities that belong to no
    single distribution: probability identities, link functions, vector
    metrics, summary statistics and combinatorics. Nothing here is
    instantiated; call the methods on the class.

    Notes
    -----
    Every method validates its arguments and raises `TypeError` or
    `ValueError` rather than returning a sentinel, except where the C++ core's
    nan-for-invalid contract is the documented behaviour.

    `sigmoid_cpu`, `logit_cpu` and the `*_cuda` methods are the batch forms,
    operating elementwise over an array. The CUDA forms raise `RuntimeError`
    when the extension was built without CUDA; check `is_cuda_available`
    first.

    This class is intended for high-performance computation using the
    underlying C++ core (`_fastdist`). Batch computations may use CPU or
    GPU (CUDA) depending on availability and input size.
    """

    @staticmethod
    def _validate_input(_input: Union[SupportsFloat, ArrayLike], input_name: str, input_type: type, dims: Union[int, None] = None) -> \
            Union[float, np.ndarray]:
        """
        Validate a scalar or sequence argument.

        Parameters
        ----------
        _input : int, float, or sequence of ints/floats
            Value(s) to validate.
        input_name : str
            Name of the argument, used in error messages.
        input_type : type
            `numbers.Real` to require a scalar, or `collections.abc.Sequence` to
            allow either a scalar or a sequence.
        dims : int or None, optional
            Required dimensionality when `_input` is array-like; 1 or 2.

        Returns
        -------
        int, float, or np.ndarray
            The validated scalar, or the sequence converted to an array.

        Raises
        ------
        TypeError
            If `_input` is None, a str or bytes, or does not match `input_type`.
        ValueError
            If `input_name` is empty, or `dims` is neither 1 nor 2, or the array
            does not have `dims` dimensions.
        """
        if _input is None:
            raise TypeError(f"{input_name} must not be None")

        # numpy reads "0.5" as data, so a string would reach the array branch
        # and come back as a one-element result instead of a TypeError.
        if isinstance(_input, (str, bytes)):
            raise TypeError(f"{input_name} must be a real number or a sequence of them")
        if isinstance(_input, Sequence) and not isinstance(_input, (str, bytes)):
            dims = 1 if dims is None else dims
        if input_name in (None, ""):
            raise ValueError("input_name must be a non-empty string")

        if input_type is int:
            if not isinstance(_input, int):
                raise TypeError(f"{input_name} must be an integer")
            return _input

        validated: Union[float, np.ndarray]
        if input_type in (Real, Sequence):
            if isinstance(_input, Real):
                validated = cast(float, _input)
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
    def _validate_array(arr: ArrayLike, arr_name: str, dims: int) -> NDArray[np.float64]:
        """
        Validate and convert a sequence to a float array.

        Parameters
        ----------
        arr : array-like
            Sequence to validate and convert.
        arr_name : str
            Name of the argument, used in error messages.
        dims : int
            Required dimensionality; 1 or 2.

        Returns
        -------
        np.ndarray
            Array of float64 with `dims` dimensions.

        Raises
        ------
        TypeError
            If the sequence contains non-numeric elements.
        ValueError
            If `arr_name` is empty, `dims` is neither 1 nor 2, or the array does
            not have `dims` dimensions.

        Notes
        -----
        Used internally by `_validate_input` so that every array reaching the C++
        core has a known dtype and shape.
        """
        if arr_name in (None, ""):
            raise ValueError("arr_name must be a non-empty string")
        if dims not in (1, 2):
            raise ValueError("dims must be 1 or 2")

        try:
            arr = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(f"{arr_name} must be numeric")

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
    def chebyshev_bound(cls, variance: SupportsFloat, k: SupportsFloat) -> float:
        """
        Compute Chebyshev's bound on a deviation from the mean.

        Bounds the probability that a random variable falls at least `k` away from
        its mean, for any distribution with finite variance:

            P(|X - mean| >= k) <= variance / k^2

        Parameters
        ----------
        variance : float
            Variance of the distribution (must be non-negative and finite).
        k : float
            Deviation from the mean, in absolute units rather than multiples of
            the standard deviation (must be positive and finite).

        Returns
        -------
        float
            The upper bound ``variance / k^2``, or nan if an argument is outside
            its domain.

        Raises
        ------
        TypeError
            If `variance` or `k` is not a real number.

        Notes
        -----
        The result is an inequality, not a probability, and is not clamped: for a
        small `k` it can exceed 1, which simply means the bound says nothing.

        To express `k` in standard deviations, pass ``k * stddev``.
        """
        cls._validate_input(_input=variance, input_name="variance", input_type=Real)
        cls._validate_input(_input=k, input_name="k", input_type=Real)
        return _core.chebyshev_bound(float(variance), float(k))

    @classmethod
    def bayes_rule(cls, p_B_given_A: SupportsFloat, p_A: SupportsFloat, p_B: SupportsFloat) -> float:
        """
        Apply Bayes' theorem to obtain a posterior probability.

            P(A|B) = P(B|A) * P(A) / P(B)

        Parameters
        ----------
        p_B_given_A : float
            Likelihood P(B|A), in [0, 1].
        p_A : float
            Prior P(A), in [0, 1].
        p_B : float
            Marginal P(B), in [0, 1] and non-zero.

        Returns
        -------
        float
            The posterior P(A|B), or nan if an argument is outside [0, 1] or
            `p_B` is zero.

        Raises
        ------
        TypeError
            If any argument is not a real number.

        Notes
        -----
        `p_B` is not derived from the other two, so it is the caller's job to keep
        the three consistent; `law_of_total_probability` computes it from a
        partition.
        """
        cls._validate_input(_input=p_B_given_A, input_name="p_B_given_A", input_type=Real)
        cls._validate_input(_input=p_A, input_name="p_A", input_type=Real)
        cls._validate_input(_input=p_B, input_name="p_B", input_type=Real)
        return _core.bayes_rule(float(p_B_given_A), float(p_A), float(p_B))

    @classmethod
    def law_of_total_probability(cls, p_A: Union[SupportsFloat, ArrayLike],
                                 p_B_given_A: Union[SupportsFloat, ArrayLike]) -> float:
        """
        Compute a marginal probability from a partition.

            P(B) = sum_i P(B|A_i) * P(A_i)

        Parameters
        ----------
        p_A : float or sequence of floats
            Prior probability of each partition element, each in [0, 1].
        p_B_given_A : float or sequence of floats
            Conditional probability of B given each partition element, each in
            [0, 1].

        Returns
        -------
        float
            The marginal P(B), or nan if any element is outside [0, 1].

        Raises
        ------
        TypeError
            If either argument is not a real number or a sequence of them.
        ValueError
            If the two arguments have different lengths.

        Notes
        -----
        A scalar is accepted as the one-element partition P(B) = P(B|A) P(A), so
        the two arguments may both be scalars.

        For the result to be a probability the `p_A` values should sum to 1; that
        is not checked, since a partial partition is a legitimate thing to sum.
        """
        # Use _validate_input to allow Real or sequence
        p_A_valid = cls._validate_input(_input=p_A, input_name="p_A", input_type=Sequence)
        p_B_given_A_valid = cls._validate_input(_input=p_B_given_A, input_name="p_B_given_A", input_type=Sequence)

        # Normalise both arguments to a list. The binding takes vectors, and a
        # scalar is the one-element partition P(B) = P(B|A) P(A) -- which the
        # signature already advertises -- so promote it rather than letting the
        # call fail inside pybind11 with an argument-type error.
        def _as_list(value: object) -> list:
            """
            Normalise a scalar, array or sequence to a plain list.

            Parameters
            ----------
            value : object
                A numpy array, a real number, or any sequence.

            Returns
            -------
            list
                The value as a list; a scalar becomes a one-element list.

            Notes
            -----
            The binding takes vectors, so a scalar is promoted here rather than
            failing inside pybind11 with an argument-type error.
            """
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Real):
                return [value]
            return list(cast(Sequence, value))

        p_A_list = _as_list(p_A_valid)
        p_B_given_A_list = _as_list(p_B_given_A_valid)

        if len(p_A_list) != len(p_B_given_A_list):
            raise ValueError("p_A and p_B_given_A must have the same length")

        return _core.law_of_total_probability(cast(Sequence[SupportsFloat], p_B_given_A_list),
                                              cast(Sequence[SupportsFloat], p_A_list))

    @classmethod
    def sigmoid(cls, x: SupportsFloat) -> float:
        """Logistic function 1 / (1 + e^-x) for a single value.

        Scalar only; use ``sigmoid_cpu`` for arrays.
        """
        # _validate_input would accept a sequence here; reject it up front so
        # the error names the array entry point instead of failing in float().
        if not isinstance(x, Real):
            raise TypeError("x must be a real number; use Utils.sigmoid_cpu for arrays")

        validated_input = cls._validate_input(_input=x, input_name="x", input_type=Real)
        return _core.sigmoid(float(validated_input))

    @classmethod
    def logit(cls, p: SupportsFloat) -> float:
        """
        Compute the logit, the inverse of the logistic function.

            logit(p) = log(p / (1 - p))

        Parameters
        ----------
        p : float
            Probability, strictly inside (0, 1).

        Returns
        -------
        float
            The log-odds of `p`, or nan if `p` is 0, 1, or outside [0, 1].

        Raises
        ------
        TypeError
            If `p` is not a real number.

        Notes
        -----
        The endpoints are excluded because the log-odds diverge there. Scalar
        only; use `logit_cpu` for arrays.
        """
        validated = cls._validate_input(_input=p, input_name="p", input_type=Real)
        return _core.logit(float(validated))

    @classmethod
    def euclidean_distance(cls, x: Sequence[SupportsFloat], y: Sequence[SupportsFloat]) -> float:
        """
        Compute the Euclidean (L2) distance between two vectors.

            sqrt(sum_i (x_i - y_i)^2)

        Parameters
        ----------
        x : sequence of floats
            First vector.
        y : sequence of floats
            Second vector, the same length as `x`.

        Returns
        -------
        float
            The distance, or nan if either vector is empty or holds a non-finite
            value.

        Raises
        ------
        ValueError
            If `x` and `y` have different lengths.
        """
        # Convert to list if numpy array; leave as-is if already Python list
        if hasattr(x, "tolist"):
            x = x.tolist()
        if hasattr(y, "tolist"):
            y = y.tolist()

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        return _core.euclidean_distance(x, y)

    @classmethod
    def manhattan_distance(cls, x: Sequence[SupportsFloat], y: Sequence[SupportsFloat]) -> float:
        """
        Compute the Manhattan (L1) distance between two vectors.

            sum_i |x_i - y_i|

        Parameters
        ----------
        x : sequence of floats
            First vector.
        y : sequence of floats
            Second vector, the same length as `x`.

        Returns
        -------
        float
            The distance, or nan if either vector is empty or holds a non-finite
            value.

        Raises
        ------
        ValueError
            If `x` and `y` have different lengths.
        """
        if hasattr(x, "tolist"):
            x = x.tolist()
        if hasattr(y, "tolist"):
            y = y.tolist()

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        return _core.manhattan_distance(x, y)

    @classmethod
    def cosine_similarity(cls, x: Sequence[SupportsFloat], y: Sequence[SupportsFloat]) -> float:
        """
        Compute the cosine similarity between two vectors.

            (x . y) / (||x|| * ||y||)

        Parameters
        ----------
        x : sequence of floats
            First vector.
        y : sequence of floats
            Second vector, the same length as `x`.

        Returns
        -------
        float
            Similarity in [-1, 1], or nan if either vector is empty, holds a
            non-finite value, or has zero norm.

        Raises
        ------
        ValueError
            If `x` and `y` have different lengths.

        Notes
        -----
        This measures orientation, not magnitude: parallel vectors give 1
        whatever their lengths.
        """
        if hasattr(x, "tolist"):
            x = x.tolist()
        if hasattr(y, "tolist"):
            y = y.tolist()

        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        return _core.cosine_similarity(x, y)

    @classmethod
    def coefficient_of_variation(cls, mean: SupportsFloat, stddev: SupportsFloat) -> float:
        """
        Compute the coefficient of variation.

            CV = stddev / |mean|

        Parameters
        ----------
        mean : float
            Mean of the distribution (must be non-zero).
        stddev : float
            Standard deviation (must be non-negative).

        Returns
        -------
        float
            The ratio, or nan if `mean` is zero or `stddev` is negative.

        Raises
        ------
        TypeError
            If either argument is not a real number.

        Notes
        -----
        Divides by the magnitude of the mean, so a negative mean gives a positive
        coefficient rather than a sign that depends on the location. The measure
        is only meaningful for a variable on a ratio scale.
        """
        cls._validate_input(_input=mean, input_name="mean", input_type=Real)
        cls._validate_input(_input=stddev, input_name="stddev", input_type=Real)
        return _core.coefficient_of_variation(float(mean), float(stddev))

    @classmethod
    def covariance(cls, mean_x: SupportsFloat, mean_y: SupportsFloat, E_xy: SupportsFloat) -> float:
        """
        Compute the population covariance from its moments.

            Cov(X, Y) = E[XY] - E[X] * E[Y]

        Parameters
        ----------
        mean_x : float
            E[X].
        mean_y : float
            E[Y].
        E_xy : float
            E[XY], the mean of the product.

        Returns
        -------
        float
            The covariance, or nan if an argument is not finite.

        Raises
        ------
        TypeError
            If any argument is not a real number.

        Notes
        -----
        This takes moments rather than samples, so it is the population
        covariance; there is no Bessel correction to apply.
        """
        cls._validate_input(_input=mean_x, input_name="mean_x", input_type=Real)
        cls._validate_input(_input=mean_y, input_name="mean_y", input_type=Real)
        cls._validate_input(_input=E_xy, input_name="E_xy", input_type=Real)
        return _core.covariance(float(mean_x), float(mean_y), float(E_xy))

    @classmethod
    def choose(cls, n: int, k: int) -> float:
        """
        Compute the binomial coefficient C(n, k).

            C(n, k) = n! / (k! * (n-k)!)

        Parameters
        ----------
        n : int
            Size of the set (must be non-negative).
        k : int
            Size of the subset (must be non-negative).

        Returns
        -------
        float
            The number of k-subsets, 0.0 if ``k > n``, or inf once the result
            exceeds the range of a double.

        Raises
        ------
        TypeError
            If `n` or `k` is not an integer.

        Notes
        -----
        Evaluated by a multiplicative recurrence over the smaller of k and n-k.
        Every partial product is itself a binomial coefficient, so each division
        is exact while the result stays within 2^53.
        """
        cls._validate_input(_input=n, input_name="n", input_type=int)
        cls._validate_input(_input=k, input_name="k", input_type=int)
        return _core.choose(n, k)

    @classmethod
    def permutation(cls, n: int, k: int) -> float:
        """
        Compute the number of ordered arrangements P(n, k).

            P(n, k) = n! / (n-k)!

        Parameters
        ----------
        n : int
            Size of the set (must be non-negative).
        k : int
            Number of items arranged (must be non-negative).

        Returns
        -------
        float
            The number of ordered k-arrangements, 0.0 if ``k > n``, or inf once
            the result exceeds the range of a double.

        Raises
        ------
        TypeError
            If `n` or `k` is not an integer.
        """
        cls._validate_input(_input=n, input_name="n", input_type=int)
        cls._validate_input(_input=k, input_name="k", input_type=int)
        return _core.permutation(n, k)

    @classmethod
    def factorial(cls, n: int) -> float:
        """
        Compute n!.

        Parameters
        ----------
        n : int
            Value whose factorial to compute (must be non-negative).

        Returns
        -------
        float
            n!, or inf for ``n > 170``.

        Raises
        ------
        TypeError
            If `n` is not an integer.

        Notes
        -----
        170! is the largest factorial a double can hold, so anything above that
        returns infinity immediately rather than iterating. Returned as a float,
        so results above 2^53 are not exact; use `log_gamma` to stay in range.
        """
        cls._validate_input(_input=n, input_name="n", input_type=int)
        return _core.factorial(n)

    @classmethod
    def gamma(cls, x: SupportsFloat) -> float:
        """
        Compute the gamma function.

            Gamma(x) = integral from 0 to inf of t^(x-1) e^-t dt

        Parameters
        ----------
        x : float
            Point at which to evaluate.

        Returns
        -------
        float
            Gamma(x), or inf once it exceeds the range of a double, or nan at the
            non-positive integers where the function has poles.

        Raises
        ------
        TypeError
            If `x` is not a real number.

        Notes
        -----
        Gamma(n+1) = n! for a non-negative integer n. Overflows between x = 171.6
        and 171.7, so Gamma(171) = 170! is the largest representable integer
        case; use `log_gamma` beyond it.
        """
        cls._validate_input(_input=x, input_name="x", input_type=Real)
        return _core.gamma(float(x))

    @classmethod
    def log_gamma(cls, x: SupportsFloat) -> float:
        """
        Compute the natural log of the gamma function.

        Parameters
        ----------
        x : float
            Point at which to evaluate.

        Returns
        -------
        float
            log Gamma(x), or nan at the non-positive integers.

        Raises
        ------
        TypeError
            If `x` is not a real number.

        Notes
        -----
        Stays finite far beyond where `gamma` overflows, which is why the
        distribution implementations build their coefficients from this rather
        than from factorials directly.
        """
        cls._validate_input(_input=x, input_name="x", input_type=Real)
        return _core.log_gamma(float(x))

    @classmethod
    def binomial(cls, n: int, a: SupportsFloat, b: SupportsFloat) -> float:
        """
        Expand (a + b)^n by the binomial theorem.

            (a + b)^n = sum_k C(n, k) * a^(n-k) * b^k

        Parameters
        ----------
        n : int
            Exponent (must be non-negative).
        a : float
            First term.
        b : float
            Second term.

        Returns
        -------
        float
            The expanded sum, which equals ``(a + b)**n`` up to rounding.

        Raises
        ------
        TypeError
            If `n` is not an integer, or `a` or `b` is not a real number.

        Notes
        -----
        Evaluates the full expansion term by term rather than taking the power
        directly, so it costs O(n^2) and overflows sooner than ``(a + b)**n``
        would. Use it to inspect the expansion, not to compute the power.
        """
        cls._validate_input(_input=n, input_name="n", input_type=int)
        cls._validate_input(_input=a, input_name="a", input_type=Real)
        cls._validate_input(_input=b, input_name="b", input_type=Real)
        return _core.binomial(n, float(a), float(b))

    # --------------------
    # Batch Static Methods
    # --------------------
    @classmethod
    def sigmoid_cpu(cls, x: ArrayLike) -> NDArray[np.float64]:
        """
        Compute the logistic sigmoid elementwise on the CPU.

        Parameters
        ----------
        x : array-like of floats
            Values at which to evaluate.

        Returns
        -------
        np.ndarray
            Array of ``1 / (1 + exp(-x))``, the same length as `x`. A non-finite
            input yields nan in its own slot.

        Raises
        ------
        TypeError
            If `x` is not array-like of real numbers.
        ValueError
            If `x` is not one-dimensional.

        Notes
        -----
        Branches on the sign of each element, so neither tail overflows.
        """
        validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=1)
        return _core.sigmoid_cpu(validated)

    @classmethod
    def logit_cpu(cls, p: ArrayLike) -> NDArray[np.float64]:
        """
        Compute the logit elementwise on the CPU.

        Parameters
        ----------
        p : array-like of floats
            Probabilities at which to evaluate, each strictly inside (0, 1).

        Returns
        -------
        np.ndarray
            Array of ``log(p / (1 - p))``, the same length as `p`. A value at or
            outside the endpoints yields nan in its own slot.

        Raises
        ------
        TypeError
            If `p` is not array-like of real numbers.
        ValueError
            If `p` is not one-dimensional.
        """
        validated = cls._validate_input(_input=p, input_name="p", input_type=Sequence, dims=1)
        return _core.logit_cpu(validated)

    # -------------------
    # CUDA Static Methods
    # -------------------
    if _CUDA_AVAILABLE:
        @classmethod
        def sigmoid_cuda(cls, x: ArrayLike) -> NDArray[np.float64]:
            """
            Compute the logistic sigmoid elementwise on the GPU.

            Parameters
            ----------
            x : array-like of floats
                Values at which to evaluate.

            Returns
            -------
            np.ndarray
                The result, the same length as `x`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If the input is not array-like of real numbers.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            validated = cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=1)
            return _core.sigmoid_cuda(validated)

        @classmethod
        def logit_cuda(cls, p: ArrayLike) -> NDArray[np.float64]:
            """
            Compute the logit elementwise on the GPU.

            Parameters
            ----------
            p : array-like of floats
                Probabilities at which to evaluate, each strictly inside (0, 1).

            Returns
            -------
            np.ndarray
                The result, the same length as `p`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If the input is not array-like of real numbers.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            validated = cls._validate_input(_input=p, input_name="p", input_type=Sequence, dims=1)
            return _core.logit_cuda(validated)

        @classmethod
        def euclidean_distance_cuda(cls, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
            """
            Compute the Euclidean (L2) distance between two vectors on the GPU.

            Parameters
            ----------
            x : array-like of floats
                First vector.
            y : array-like of floats
                Second vector, the same length as `x`.

            Returns
            -------
            float
                The computed value.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            ValueError
                If `x` and `y` have different lengths.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            x_validated = cast(np.ndarray, cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=2))
            y_validated = cast(np.ndarray, cls._validate_input(_input=y, input_name="y", input_type=Sequence, dims=2))

            if x_validated.shape != y_validated.shape:
                raise ValueError("x and y must have the same shape")

            return _core.euclidean_distance_cuda(x_validated, y_validated)

        @classmethod
        def manhattan_distance_cuda(cls, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
            """
            Compute the Manhattan (L1) distance between two vectors on the GPU.

            Parameters
            ----------
            x : array-like of floats
                First vector.
            y : array-like of floats
                Second vector, the same length as `x`.

            Returns
            -------
            float
                The computed value.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            ValueError
                If `x` and `y` have different lengths.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            x_validated = cast(np.ndarray, cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=2))
            y_validated = cast(np.ndarray, cls._validate_input(_input=y, input_name="y", input_type=Sequence, dims=2))

            if x_validated.shape != y_validated.shape:
                raise ValueError("x and y must have the same shape")

            return _core.manhattan_distance_cuda(x_validated, y_validated)

        @classmethod
        def cosine_similarity_cuda(cls, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
            """
            Compute the cosine similarity between two vectors on the GPU.

            Parameters
            ----------
            x : array-like of floats
                First vector.
            y : array-like of floats
                Second vector, the same length as `x`.

            Returns
            -------
            float
                The computed value.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            ValueError
                If `x` and `y` have different lengths.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            x_validated = cast(np.ndarray, cls._validate_input(_input=x, input_name="x", input_type=Sequence, dims=2))
            y_validated = cast(np.ndarray, cls._validate_input(_input=y, input_name="y", input_type=Sequence, dims=2))

            if x_validated.shape != y_validated.shape:
                raise ValueError("x and y must have the same shape")

            return _core.cosine_similarity_cuda(x_validated, y_validated)
    else:
        @classmethod
        def sigmoid_cuda(cls, *args, **kwargs):
            """
            Compute the logistic sigmoid elementwise on the GPU.

            Parameters
            ----------
            x : array-like of floats
                Values at which to evaluate.

            Returns
            -------
            np.ndarray
                The result, the same length as `x`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If the input is not array-like of real numbers.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def logit_cuda(cls, *args, **kwargs):
            """
            Compute the logit elementwise on the GPU.

            Parameters
            ----------
            p : array-like of floats
                Probabilities at which to evaluate, each strictly inside (0, 1).

            Returns
            -------
            np.ndarray
                The result, the same length as `p`.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            TypeError
                If the input is not array-like of real numbers.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def euclidean_distance_cuda(cls, *args, **kwargs):
            """
            Compute the Euclidean (L2) distance between two vectors on the GPU.

            Parameters
            ----------
            x : array-like of floats
                First vector.
            y : array-like of floats
                Second vector, the same length as `x`.

            Returns
            -------
            float
                The computed value.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            ValueError
                If `x` and `y` have different lengths.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def manhattan_distance_cuda(cls, *args, **kwargs):
            """
            Compute the Manhattan (L1) distance between two vectors on the GPU.

            Parameters
            ----------
            x : array-like of floats
                First vector.
            y : array-like of floats
                Second vector, the same length as `x`.

            Returns
            -------
            float
                The computed value.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            ValueError
                If `x` and `y` have different lengths.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )

        @classmethod
        def cosine_similarity_cuda(cls, *args, **kwargs):
            """
            Compute the cosine similarity between two vectors on the GPU.

            Parameters
            ----------
            x : array-like of floats
                First vector.
            y : array-like of floats
                Second vector, the same length as `x`.

            Returns
            -------
            float
                The computed value.

            Raises
            ------
            RuntimeError
                If the extension was built without CUDA support.
            ValueError
                If `x` and `y` have different lengths.

            Notes
            -----
            Check `is_cuda_available` before calling. The CPU form is faster below
            roughly 100,000 elements, where the transfer dominates.
            """
            raise RuntimeError(
                "CUDA support is not available. This package was built without CUDA. "
                "Please use CPU methods or reinstall with CUDA support."
            )
