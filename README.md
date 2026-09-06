# fastdist

High-performance probability distributions and statistical functions for Python, backed by a C++ core with
optional CUDA acceleration.

[![Build Wheels](https://github.com/ghosteau/fastdist/actions/workflows/wheels.yml/badge.svg)](https://github.com/ghosteau/fastdist/actions/workflows/wheels.yml)
[![Python Distro Check](https://github.com/ghosteau/fastdist/actions/workflows/python-distro.yml/badge.svg)](https://github.com/ghosteau/fastdist/actions/workflows/python-distro.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/ghosteau/fastdist/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg)](https://www.python.org/downloads/)

> **Alpha.** The API is still moving and there are known gaps — see [Status](#status).

---

## Installation

> **Not on PyPI yet.** The name `fastdist` there belongs to an unrelated distance-metrics library, so
> `pip install fastdist` will **not** get this project. Build from a checkout for now.

```bash
git clone https://github.com/ghosteau/fastdist.git
cd fastdist
pip install .
```

You need a C++20 compiler and CMake 3.20+. CMake is pulled in automatically as a build dependency, so on
most systems `pip install .` is all that is required.

---

## Quickstart

```python
import numpy as np
from fastdist import Normal, Binomial, Utils

# Continuous distributions take scalars or arrays
n = Normal(mu=0.0, sigma=1.0)
n.pdf(0.0)                          # 0.39894228040143265
n.pdf(np.array([-1.0, 0.0, 1.0]))   # array([0.24197072, 0.39894228, 0.24197072])
n.cdf(1.96)                         # 0.9750021048517796

n.mean(), n.variance(), n.stddev()  # (0.0, 1.0, 1.0)
n.z_score(2.0)                      # 2.0
n.sample()                          # a single draw

# Discrete distributions expose the same moments
b = Binomial(n=10, p=0.3)
b.pmf_scalar(3)                     # 0.266827932
b.cdf_scalar(3)                     # 0.6496107184
b.mean(), b.variance()              # (3.0, 2.0999999999999996)

# Statistical helpers
Utils.sigmoid(0.5)                                              # 0.6224593312018546
Utils.choose(10, 3)                                             # 120.0
Utils.euclidean_distance(np.array([0., 0.]), np.array([3., 4.]))  # 5.0
Utils.chebyshev_bound(variance=4.0, k=3.0)                      # 0.444...
```

### Scalar and array APIs

Not every distribution is vectorised yet. Where a CUDA path exists the class exposes array-capable
`pdf`/`pmf`/`cdf`, which dispatch to CPU or GPU depending on input size:

| Vectorised (`pdf` / `pmf` / `cdf`) | Scalar only (`*_scalar`) |
|---|---|
| Normal (also `logpdf`), Bernoulli, Poisson, Uniform, Exponential, ChiSquare, DiscreteUniform | Binomial, Beta, Gamma, Geometric, NegativeBinomial |

Every distribution provides `mean()`, `variance()`, `stddev()`, `sample()`, and MGF/CGF where one exists in
closed form.

---

## What's included

**Distributions** — Bernoulli, Beta, Binomial, Chi-square, Discrete Uniform, Exponential, Gamma, Geometric,
Negative Binomial, Normal, Poisson, Uniform.

All provide PDF/PMF and CDF. Moment-generating and cumulant-generating functions are available for the
subset with closed forms.

**Statistical utilities** — Chebyshev's inequality, Bayes' theorem, law of total probability, sigmoid and
logit, Euclidean/Manhattan distance, cosine similarity, coefficient of variation, covariance, combinatorics
(`choose`, `factorial`, `permutation`), and gamma / log-gamma.

---

## CUDA

CUDA support is **early stage** and off by default; the published wheels are CPU-only. Five distributions
have GPU kernels — Normal, Uniform, Exponential, Bernoulli, Poisson — along with the distance and
sigmoid/logit utilities.

To build with CUDA you need an NVIDIA GPU, the CUDA toolkit, and on Windows a Visual Studio installation:

```bash
FASTDIST_ENABLE_CUDA=1 pip install .
```

Where a GPU path exists, calls above a size threshold dispatch to it automatically. Thresholds are tunable:

```python
from fastdist import config

config.get_cuda_threshold("normal_pdf")
config.set_cuda_threshold("normal_pdf", 250_000)
config.auto_tune(classes=["normal"])   # benchmark CPU vs GPU and persist the crossover
```

`auto_tune` writes to `~/.config/fastdist/config.json` (`%LOCALAPPDATA%\fastdist\config.json` on Windows).

---

## Status

This is a pre-1.0 library and the following are known:

- CUDA covers 5 of 12 distributions.

---

## Contributing

Build instructions, project layout, how to add a distribution, and the test and formatting workflow are in
[CONTRIBUTING.md](https://github.com/ghosteau/fastdist/blob/main/CONTRIBUTING.md).

Release history is in [CHANGELOG.md](https://github.com/ghosteau/fastdist/blob/main/CHANGELOG.md). To report a vulnerability, see
[SECURITY.md](https://github.com/ghosteau/fastdist/blob/main/SECURITY.md).

---

## License

Apache License 2.0 — see [LICENSE](https://github.com/ghosteau/fastdist/blob/main/LICENSE).

## Authors

- Manny McGrail
- Zach Pipes
