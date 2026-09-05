# python/distributions/gamma.py

try:
    from .. import _fastdist as _core
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

class Gamma:
    # Magic Methods
    __slots__ = ("_alpha", "_theta")

    def __init__(self, alpha: Union[int, float], theta: Union[int, float]):
        self._validate_params(alpha=alpha, theta=theta)
        self._alpha = float(alpha)
        self._theta = float(theta)

    @property
    def alpha(self):
        return self._alpha

    @property
    def theta(self):
        return self._theta

    @alpha.setter
    def alpha(self, value):
        self._validate_params(alpha=value)
        self._alpha = float(value)

    @theta.setter
    def theta(self, value):
        self._validate_params(theta=value)
        self._theta = float(value)

    def __repr__(self):
        return f"Gamma(alpha={self.alpha}, theta={self.theta})"

    @staticmethod
    def _validate_params(alpha: Union[int, float] = None, theta: Union[int, float] = None) -> None:
        """Internal validation shared by all methods."""
        if alpha is not None:
            if not isinstance(alpha, (int, float)):
                raise TypeError("alpha must be a real number")
            if alpha <= 0:
                raise ValueError("alpha must be positive")
        if theta is not None:
            if not isinstance(theta, (int, float)):
                raise TypeError("theta must be a real number")
            if theta <= 0:
                raise ValueError("theta must be positive")

    @staticmethod
    def _validate_inputs(x=None, t=None) -> None:
        if x is not None and not isinstance(x, (int, float)):
            raise TypeError("x must be a real number")
        if t is not None and not isinstance(t, (int, float)):
            raise TypeError("t must be a real number")

    # ------------------------------------------------------------------------------------------------------------------
    # Instance Methods
    # ------------------------------------------------------------------------------------------------------------------
    def pmf_scalar(self, x: Union[int, float]) -> float:
        self._validate_inputs(x=x)
        return _core.gamma_pdf_scalar(x, self.alpha, self.theta)

    def cdf_scalar(self, x: Union[int, float]) -> float:
        self._validate_inputs(x=x)
        return _core.gamma_cdf_scalar(x, self.alpha, self.theta)

    def mean(self) -> float:
        return _core.gamma_mean(self.alpha, self.theta)

    def variance(self) -> float:
        return _core.gamma_variance(self.alpha, self.theta)

    def stddev(self) -> float:
        return _core.gamma_stddev(self.alpha, self.theta)

    def mgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.gamma_mgf_scalar(t, self.alpha, self.theta)

    def cgf_scalar(self, t: Union[int, float]) -> float:
        self._validate_inputs(t=t)
        return _core.gamma_cgf_scalar(t, self.alpha, self.theta)

    def sample(self) -> float:
        return _core.gamma_sample(self.alpha, self.theta)

    # ------------------------------------------------------------------------------------------------------------------
    # Scalar Static Methods
    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _pdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(x=x)
        return _core.gamma_pdf_scalar(float(x), float(alpha), float(theta))

    @classmethod
    def _cdf_scalar(cls, x: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(x=x)
        return _core.gamma_cdf_scalar(float(x), float(alpha), float(theta))

    @classmethod
    def _mean(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_mean(float(alpha), float(theta))

    @classmethod
    def _variance(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_variance(float(alpha), float(theta))

    @classmethod
    def _stddev(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_stddev(float(alpha), float(theta))

    @classmethod
    def _mgf_scalar(cls, t: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(t=t)
        return _core.gamma_mgf_scalar(float(t), float(alpha), float(theta))

    @classmethod
    def _cgf_scalar(cls, t: Union[int, float], alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        cls._validate_inputs(t=t)
        return _core.gamma_cgf_scalar(float(t), float(alpha), float(theta))

    @classmethod
    def _sample(cls, alpha: Union[int, float], theta: Union[int, float]) -> float:
        cls._validate_params(alpha=alpha, theta=theta)
        return _core.gamma_sample(float(alpha), float(theta))
