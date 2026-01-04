# python/distributions/utils.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Utils:
    @staticmethod
    def _validate_int(value, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    @staticmethod
    def _validate_real(value, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")

    @staticmethod
    def _validate_real_sequence(seq, name: str) -> list[float]:
        if not isinstance(seq, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple of real numbers")
        if not seq:
            raise ValueError(f"{name} must not be empty")
        result = []
        for i, v in enumerate(seq):
            if not isinstance(v, (int, float)):
                raise TypeError(f"{name}[{i}] must be a real number")
            result.append(float(v))
        return result

    @classmethod
    def _chebyshev_bound(cls, variance: int | float, k: int | float) -> float:
        cls._validate_real(variance, "variance")
        cls._validate_real(k, "k")
        return _core.chebyshev_bound(float(variance), float(k))

    @classmethod
    def _bayes_rule(cls, p_B_given_A: int | float, p_A: int | float, p_B: int | float) -> float:
        cls._validate_real(p_B_given_A, "p_B_given_A")
        cls._validate_real(p_A, "p_A")
        cls._validate_real(p_B, "p_B")
        return _core.bayes_rule(float(p_B_given_A), float(p_A), float(p_B))

    @classmethod
    def _law_of_total_probability(cls, p_A: int | float, p_B_given_A: int | float) -> float:
        cls._validate_real(p_A, "p_A")
        cls._validate_real(p_B_given_A, "p_B_given_A")
        return _core.law_of_total_probability(float(p_A), float(p_B_given_A))

    @classmethod
    def _sigmoid(cls, x: int | float) -> float:
        cls._validate_real(x, "x")
        return _core.sigmoid(float(x))

    @classmethod
    def _logit(cls, p: int | float) -> float:
        cls._validate_real(p, "p")
        return _core.logit(float(p))

    @classmethod
    def _euclidean_distance(cls, x: list[float] | tuple[float, ...], y: list[float] | tuple[float, ...]) -> float:
        x_floats = cls._validate_real_sequence(x, "x")
        y_floats = cls._validate_real_sequence(y, "y")
        return _core.euclidean_distance(x_floats, y_floats)

    @classmethod
    def _manhattan_distance(cls, x: list[float] | tuple[float, ...], y: list[float] | tuple[float, ...]) -> float:
        x_floats = cls._validate_real_sequence(x, "x")
        y_floats = cls._validate_real_sequence(y, "y")
        return _core.manhattan_distance(x_floats, y_floats)

    @classmethod
    def _cosine_similarity(cls, x: list[float] | tuple[float, ...], y: list[float] | tuple[float, ...]) -> float:
        x_floats = cls._validate_real_sequence(x, "x")
        y_floats = cls._validate_real_sequence(y, "y")
        return _core.cosine_similarity(x_floats, y_floats)

    @classmethod
    def _coefficient_of_variation(cls, mean: int | float, stddev: int | float) -> float:
        cls._validate_real(mean, "mean")
        cls._validate_real(stddev, "stddev")
        return _core.coefficient_of_variation(float(mean), float(stddev))

    @classmethod
    def _covariance(cls, mean_x: int | float, mean_y: int | float, E_xy: int | float) -> float:
        cls._validate_real(mean_x, "mean_x")
        cls._validate_real(mean_y, "mean_y")
        cls._validate_real(E_xy, "E_xy")
        return _core.covariance(float(mean_x), float(mean_y), float(E_xy))

    @classmethod
    def _choose(cls, n: int, k: int) -> float:
        cls._validate_int(n, "n")
        cls._validate_int(k, "k")
        return _core.choose(n, k)

    @classmethod
    def _permutation(cls, n: int, k: int) -> float:
        cls._validate_int(n, "n")
        cls._validate_int(k, "k")
        return _core.permutation(n, k)

    @classmethod
    def _factorial(cls, n: int) -> float:
        cls._validate_int(n, "n")
        return _core.factorial(n)

    @classmethod
    def _gamma(cls, x: int | float) -> float:
        cls._validate_real(x, "x")
        return _core.gamma(float(x))

    @classmethod
    def _log_gamma(cls, x: int | float) -> float:
        cls._validate_real(x, "x")
        return _core.log_gamma(float(x))

    @classmethod
    def _binomial(cls, n: int, a: int | float, b: int | float) -> float:
        cls._validate_int(n, "n")
        cls._validate_real(a, "a")
        cls._validate_real(b, "b")
        return _core.binomial(n, float(a), float(b))
