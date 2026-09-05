# python/fastdist/__init__.py
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import _fastdist
from ._fastdist import seed, seed_from_entropy
from .distributions import (
    Bernoulli, Beta, Binomial, ChiSquare, DiscreteUniform,
    Exponential, Gamma, Geometric, NegativeBinomial, Normal,
    Poisson, Uniform, Utils
)

try:
    __version__ = _pkg_version("fastdist")
except PackageNotFoundError:  # running from an uninstalled source checkout
    __version__ = "0.0.0+unknown"

__all__ = ["__version__",
           "Bernoulli", "Beta", "Binomial", "ChiSquare",
           "DiscreteUniform", "Exponential", "Gamma",
           "Geometric", "NegativeBinomial", "Normal",
           "Poisson", "Uniform", "Utils",
           "seed", "seed_from_entropy"]