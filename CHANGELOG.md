# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version is defined once, in the `project()`
call in `CMakeLists.txt`.

---

## [Unreleased]

### Fixed

- `binomial_cdf_scalar`, `poisson_cdf_scalar` and `negative_binomial_cdf_scalar` returned the raw sum of
  PMF terms, which accumulates rounding error and could exceed `1.0`. For the binomial this also made the
  CDF non-monotonic, because `x >= n` short-circuits to exactly `1.0` while the values just below it did
  not. All three now clamp to `1.0`.
- `setup.py` no longer hardcodes the `Visual Studio 17 2022` CMake generator. CMake selects the newest
  Visual Studio present, so builds work on machines with a different version installed. Set
  `CMAKE_GENERATOR` to pin one.

### Added

- Cross-platform wheel building in CI via `cibuildwheel` — Linux x86_64, Windows AMD64, and macOS
  x86_64 + arm64, for CPython 3.10 through 3.14.
- An install-from-sdist check in CI, exercising the source path an end user takes on any platform without
  a prebuilt wheel.
- `MANIFEST.in`, so the source distribution actually contains the C++ sources, headers and `CMakeLists.txt`
  it needs to build. Previously the sdist shipped Python files only and could not be built from.
- CTest integration: `enable_testing()` plus one test entry per C++ test file, so failures are isolated and
  `ctest -R <name>` works. `fastdist_tests` accepts test names as arguments.
- `py.typed`, so type checkers honour the package's annotations.
- `CONTRIBUTING.md`, `CHANGELOG.md` and `SECURITY.md`.

### Changed

- `NDEBUG` is undefined for the `fastdist_tests` target, so its `assert()`-based checks stay live in Release
  builds. They were previously compiled away, meaning the suite reported success without testing anything.
- Repository layout: bindings moved from `python/bindings` to `src/bindings`; tests split into `tests/cpp`
  and `tests/python`; the showcase notebook moved to `examples/`.
- `python_requires` lowered to `>=3.10` and reconciled with the CMake Python requirement, which previously
  disagreed with it.
- `cmake` and `ninja` are declared as build dependencies, so an install from source no longer requires
  CMake to be installed beforehand.

### Known issues

- `negative_binomial_pmf_scalar` returns `inf` for `k` around 169 and `NaN` beyond it, because the binomial
  coefficient is computed with `std::tgamma`, which overflows above ~171. Computing it in log space via
  `std::lgamma` is the fix. `beta.cpp` and `gamma.cpp` use `tgamma` similarly.
- Three RNG test assertions have tolerances at roughly 2σ and fail a few percent of runs. CI absorbs this
  with `ctest --repeat until-pass:3`.
- The samplers seed `std::mt19937` from a single 32-bit `random_device` word.

---

## [0.1.0] — 2026-09-01

Initial pre-release, establishing the core architecture, API surface and build system.

### Added

- **Distributions** — Bernoulli, Beta, Binomial, Chi-square, Discrete Uniform, Exponential, Gamma,
  Geometric, Negative Binomial, Normal, Poisson, Uniform. All provide PDF/PMF and CDF; MGFs are available
  for the subset with closed forms.
- **Statistical utilities** — Chebyshev's inequality, Bayes' theorem, law of total probability, sigmoid and
  logit, Euclidean and Manhattan distance, cosine similarity, coefficient of variation, covariance,
  combinatorics (`choose`, `factorial`, binomial theorem), and gamma / log-gamma.
- **CUDA backend (early support)** — GPU-accelerated computation and RNG, with clustered and batched
  workflows. Covers Normal, Uniform, Exponential and Bernoulli, plus the distance, cosine-similarity,
  sigmoid and logit utilities.
- **Python bindings** — pybind11 resolved from the build environment.
- **Testing and CI** — unit tests covering core functionality, with GitHub Actions pipelines.

[Unreleased]: https://github.com/ghosteau/fastdist/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ghosteau/fastdist/releases/tag/v0.1.0
