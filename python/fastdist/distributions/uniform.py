# python/distributions/uniform.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")


class Uniform:
    # Magic Methods
    __slots__ = ("a", "b")

    def __init__(self, a: int | float, b: int | float):
        self._validate_params(a=a, b=b)
        self.a = float(a)
        self.b = float(b)

    def __repr__(self):
        return f"Uniform(a={self.a}, b={self.b})"

    @staticmethod
    def _validate_params(a: int | float, b: int | float) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(a, (int, float)):
            raise TypeError("a must be a real number")
        if not isinstance(b, (int, float)):
            raise TypeError("b must be a real number")
        if a >= b:
            raise ValueError("a must be less than b")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # Instance Methods
    def pdf_scalar(self, x: int | float) -> float:
        self._validate_inputs(x=x)
        return _core.uniform_pdf_scalar(float(x), self.a, self.b)

    def cdf_scalar(self, x: int | float) -> float:
        self._validate_inputs(x=x)
        return _core.uniform_cdf_scalar(float(x), self.a, self.b)

    def mean(self) -> float:
        return _core.uniform_mean(self.a, self.b)

    def variance(self) -> float:
        return _core.uniform_variance(self.a, self.b)

    def stddev(self) -> float:
        return _core.uniform_stddev(self.a, self.b)

    def mgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.uniform_mgf_scalar(float(t), self.a, self.b)

    def cgf_scalar(self, t: int | float) -> float:
        self._validate_inputs(t=t)
        return _core.uniform_cgf_scalar(float(t), self.a, self.b)

    def sample(self) -> float:
        return _core.uniform_sample(self.a, self.b)

    # Static Methods
    @classmethod
    def _pdf_scalar(cls, x: int | float, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(x=x)
        return _core.uniform_pdf_scalar(float(x), float(a), float(b))

    @classmethod
    def _cdf_scalar(cls, x: int | float, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(x=x)
        return _core.uniform_cdf_scalar(float(x), float(a), float(b))

    @classmethod
    def _mean(cls, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        return _core.uniform_mean(float(a), float(b))

    @classmethod
    def _variance(cls, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        return _core.uniform_variance(float(a), float(b))

    @classmethod
    def _stddev(cls, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        return _core.uniform_stddev(float(a), float(b))

    @classmethod
    def _mgf_scalar(cls, t: int | float, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(t=t)
        return _core.uniform_mgf_scalar(float(t), float(a), float(b))

    @classmethod
    def _cgf_scalar(cls, t: int | float, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(t=t)
        return _core.uniform_cgf_scalar(float(t), float(a), float(b))

    @classmethod
    def _sample(cls, a: int | float, b: int | float) -> float:
        cls._validate_params(a=a, b=b)
        return _core.uniform_sample(float(a), float(b))
