# python/distributions/chi_square.py
try:
    from fastdist import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

from . import Real, Sequence, Union, NDArray

class ChiSquare:
    # Magic Methods
    __slots__ = ("_k",)

    def __init__(self, k: Union[int, float]):
        ChiSquare._validate_params(k=k)
        self._k = float(k)

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._validate_params(k=value)
        self._k = float(value)

    def __repr__(self):
        return f"ChiSquare(k={self.k})"

    @staticmethod
    def _validate_params(k: Union[int, float]) -> None:
        """Internal validation shared by all methods."""
        if not isinstance(k, (int, float)):
            raise TypeError("k must be a real number")
        if k <= 0:
            raise ValueError("k must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pdf(self, x: Union[int, float]) -> float:
        self._validate_inputs(x=x)
        return _core.chi_square_pdf_scalar(float(x), self.k)

    def cdf(self, x: Union[int, float]) -> float:
        self._validate_inputs(x=x)
        return _core.chi_square_cdf_scalar(float(x), self.k)

    def mean(self) -> float:
        return _core.chi_square_mean(self.k)

    def variance(self) -> float:
        return _core.chi_square_variance(self.k)

    def stddev(self) -> float:
        return _core.chi_square_stddev(self.k)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.chi_square_mgf_scalar(float(t), self.k)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.chi_square_cgf_scalar(float(t), self.k)

    def sample(self) -> float:
        return _core.chi_square_sample(self.k)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float], k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        cls._validate_inputs(x=x)
        return _core.chi_square_pdf_scalar(float(x), float(k))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float], k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        cls._validate_inputs(x=x)
        return _core.chi_square_cdf_scalar(float(x), float(k))

    @classmethod
    def _mean(cls, k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        return _core.chi_square_mean(float(k))

    @classmethod
    def _variance(cls, k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        return _core.chi_square_variance(float(k))

    @classmethod
    def _stddev(cls, k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        return _core.chi_square_stddev(float(k))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        cls._validate_inputs(t=t)
        return _core.chi_square_mgf_scalar(float(t), float(k))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        cls._validate_inputs(t=t)
        return _core.chi_square_cgf_scalar(float(t), float(k))

    @classmethod
    def _sample(cls, k: Union[int, float]) -> float:
        cls._validate_params(k=k)
        return _core.chi_square_sample(float(k))
