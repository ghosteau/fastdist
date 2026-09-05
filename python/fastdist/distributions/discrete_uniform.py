# python/distributions/discrete_uniform.py
try:
    from fastdist import _fastdist as _core
except ImportError as exc:  # pragma: no cover - only hit in a broken install
    raise ImportError(
        "fastdist's compiled extension (_fastdist) could not be imported. "
        "Build it with `pip install .` from the repository root; importing "
        "the package straight from a source checkout will not work until the "
        "extension has been built."
    ) from exc

import numpy as np
from numbers import Real
from typing import Sequence, Union
from numpy.typing import NDArray

class DiscreteUniform:
    # Magic Methods
    __slots__ = ("_a", "_b")

    def __init__(self, a: int, b: int):
        self._validate_params(a=a, b=b)
        self._a = int(a)
        self._b = int(b)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @a.setter
    def a(self, value):
        self._validate_params(a=value)
        self._a = float(value)

    @b.setter
    def b(self, value):
        self._validate_params(b=value)
        self.b = value

    def __repr__(self):
        return f"DiscreteUniform(a={self.a}, b={self.b})"

    @staticmethod
    def _validate_params(a: int = None, b: int = None) -> None:
        """Internal validation shared by all methods."""
        if a is not None:
            if not isinstance(a, int):
                raise TypeError("a must be an integer")
        if b is not None:
            if not isinstance(b, int):
                raise TypeError("b must be an integer")
        if a is not None and b is not None:
            if a >= b:
                raise ValueError("a must be less than b")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        if x is not None and not isinstance(x, int):
            raise TypeError("x must be an integer")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pmf(self, x: int) -> float:
        self._validate_inputs(x=x)
        return _core.discrete_uniform_pmf_scalar(x, self.a, self.b)

    def cdf(self, x: int) -> float:
        self._validate_inputs(x=x)
        return _core.discrete_uniform_cdf_scalar(x, self.a, self.b)

    def mean(self) -> float:
        return _core.discrete_uniform_mean(self.a, self.b)

    def variance(self) -> float:
        return _core.discrete_uniform_variance(self.a, self.b)

    def stddev(self) -> float:
        return _core.discrete_uniform_stddev(self.a, self.b)

    def mgf(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.discrete_uniform_mgf_scalar(float(t), self.a, self.b)

    def cgf(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.discrete_uniform_cgf_scalar(float(t), self.a, self.b)

    def sample(self) -> int:
        return _core.discrete_uniform_sample(self.a, self.b)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pmf_scalar(cls, x: int, a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(x=x)
        return _core.discrete_uniform_pmf_scalar(x, a, b)

    @classmethod
    def _cdf_scalar(cls, x: int, a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(x=x)
        return _core.discrete_uniform_cdf_scalar(x, a, b)

    @classmethod
    def _mean(cls, a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_mean(a, b)

    @classmethod
    def _variance(cls, a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_variance(a, b)

    @classmethod
    def _stddev(cls, a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_stddev(a, b)

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(t=t)
        return _core.discrete_uniform_mgf_scalar(float(t), a, b)

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], a: int, b: int) -> float:
        cls._validate_params(a=a, b=b)
        cls._validate_inputs(t=t)
        return _core.discrete_uniform_cgf_scalar(float(t), a, b)

    @classmethod
    def _sample(cls, a: int, b: int) -> int:
        cls._validate_params(a=a, b=b)
        return _core.discrete_uniform_sample(a, b)
