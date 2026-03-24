# python/distributions/bernoulli.py
try:
    from .. import _fastdist as _core
except ImportError:
    raise ImportError("Internal Error: C++ core (_fastdist) not found. Check package structure.")

import numpy as np
from numbers import Real
from typing import Sequence, Union
from numpy.typing import NDArray

class Beta:
    # Magic Methods
    __slots__ = ("_alpha", "_beta")

    def __init__(self, alpha: Union[int, float], beta: Union[int, float]):
        self._validate_params(alpha=alpha, beta=beta)
        self._alpha = float(alpha)
        self._beta = float(beta)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @alpha.setter
    def alpha(self, value):
        self._validate_params(alpha=value)
        self._alpha = float(value)

    @beta.setter
    def beta(self, value):
        self._validate_params(beta=value)
        self._beta = float(value)

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta})"

    @staticmethod
    def _validate_params(alpha: Union[int, float] = None, beta: Union[int, float] = None) -> None:
        """Internal validation shared by all methods."""
        if alpha is not None:
            if not isinstance(alpha, (int, float)):
                raise TypeError("alpha must be a real number")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if beta is not None:
            if not isinstance(beta, (int, float)):
                raise TypeError("beta must be a real number")
            if beta <= 0:
                raise ValueError("beta must be positive")

    @staticmethod
    def _validate_inputs(x=None) -> None:
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pdf_scalar(self, x: Union[int, float]) -> float:
        self._validate_inputs(x=x)
        return _core.beta_pdf_scalar(float(x), self.alpha, self.beta)

    def cdf_scalar(self, x: Union[int, float]) -> float:
        self._validate_inputs(x=x)
        return _core.beta_cdf_scalar(float(x), self.alpha, self.beta)

    def mean(self) -> float:
        return _core.beta_mean(self.alpha, self.beta)

    def variance(self) -> float:
        return _core.beta_variance(self.alpha, self.beta)

    def stddev(self) -> float:
        return _core.beta_stddev(self.alpha, self.beta)

    def sample(self) -> float:
        return _core.beta_sample(self.alpha, self.beta)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], beta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, beta=beta)
        cls._validate_inputs(x=x)
        return _core.beta_pdf_scalar(float(x), float(alpha), float(beta))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], beta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, beta=beta)
        cls._validate_inputs(x=x)
        return _core.beta_cdf_scalar(float(x), float(alpha), float(beta))

    @classmethod
    def _mean(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_mean(float(alpha), float(beta))

    @classmethod
    def _variance(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_variance(float(alpha), float(beta))

    @classmethod
    def _stddev(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_stddev(float(alpha), float(beta))

    @classmethod
    def _sample(cls, alpha: Union[int, float], beta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, beta=beta)
        return _core.beta_sample(float(alpha), float(beta))
