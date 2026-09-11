# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The version is defined once, in the `project()`
call in `CMakeLists.txt`.

---

## [Unreleased]

### Fixed

- `beta_cdf_scalar` was wrong at every point, not only at the extremes: 0.0015 against a true 0.1143 for
  Beta(2, 5) at x = 0.1, and values outside [0, 1] such as -147 for Beta(0.01, 0.01) at x = 0.5. It is now
  the modified-Lentz continued fraction with the standard reflection, and agrees with SciPy to ~1e-12.
- `gamma_cdf_scalar` and `chi_square_cdf_scalar` returned probabilities above 1.0 (Gamma(1.5, 1).cdf(2.5)
  gave 1.000498). A unary minus applied to an unsigned loop index wrapped to 2^32 - i inside the continued
  fraction. Separately, the series stopped at 100 iterations and silently truncated for large shapes (off
  by 0.16 at alpha = 10000); the ceiling is now 1000.
- `beta_pdf_scalar`, `gamma_pdf_scalar` and `chi_square_pdf_scalar` returned `nan` or `inf` once a shape
  parameter passed ~171 (k above ~342 for chi-square), because the normalising constants overflowed
  `std::tgamma`. All three are now evaluated in log space.
- `negative_binomial_pmf_scalar` returned `inf` at k = 170 and `nan` beyond it, taking the CDF with it, for
  the same reason. It is now evaluated in log space.
- `binomial_cdf_scalar`, `poisson_cdf_scalar` and `negative_binomial_cdf_scalar` could exceed `1.0`
  through accumulated rounding, which also made the binomial CDF non-monotonic. All three now clamp to
  `1.0`.
- `beta_sample` did not validate its parameters, and `std::gamma_distribution` has undefined behaviour for
  a non-positive shape. It now returns `nan` for invalid input, like every other continuous sampler.
- Python setters: `Beta.beta` and `Binomial.p` raised `TypeError` for every value, valid or not;
  `DiscreteUniform.b` recursed until `RecursionError`; `DiscreteUniform.a` changed the attribute's type to
  `float`; and the `Uniform` and `DiscreteUniform` setters accepted a bound that violated `a < b`, after
  which every method silently returned `nan`.
- `Utils.law_of_total_probability` rejected scalar arguments despite its signature. Scalars are now
  treated as a one-element partition, and sequences of different lengths raise `ValueError`.
- An `ImportError` raised while loading a distribution module, such as a missing numpy, was reported as a
  missing C++ core. The original exception is now chained, and the message says how to build the
  extension.
- The C++ RNG tests failed in about 7.7% of runs because their tolerances sat near 2σ of the estimator's
  own noise (#2).
- The CUDA backend did not compile on Windows. `nvcc` 12.x's front end crashes on MSVC's C++20
  standard-library headers, and every `.cu` file was compiled a second time into the Python module
  target, which built as C++20. CUDA sources are now compiled once, as C++17.
- The CUDA extension could not be imported on Windows, because it depended on `cudart64_*.dll` and
  Python does not search `PATH` for extension dependencies. Once imported, its first GPU call crashed
  with an access violation, because the wrapper released the GIL before touching the input and output
  arrays. The CUDA runtime is now linked statically, and the GIL is released only around the device
  work.
- `normal_cdf` lost all relative precision in the lower tail: Phi(-8) was off by 1.8%, and Phi(-10)
  returned exactly 0 instead of 7.6e-24. `exponential_cdf` did the same for small arguments,
  returning 0 at x = 1e-17. Both now use `erfc` and `expm1` respectively, on the CPU and GPU paths.
- `setup.py` no longer hardcodes the `Visual Studio 17 2022` CMake generator. CMake selects the newest
  Visual Studio present, so builds work on machines with a different version installed. Set
  `CMAKE_GENERATOR` to pin one.

### Added

- `fastdist.seed(value)` and `fastdist.seed_from_entropy()`, with C++ equivalents `seed_rng` and
  `seed_rng_from_entropy` in `fastdist/math/rng.h`. Every sampler now draws from one shared thread-local
  Mersenne Twister, seeded through `std::seed_seq`, and any signed 64-bit value is accepted. A seed
  reproduces a run on one platform and toolchain, and applies to the calling thread only.
- A benchmark suite under `benchmarks/`, and `BENCHMARKS.md` as a performance log generated from its
  recorded results. It compares against SciPy, checks numerical agreement before timing, flags regressions
  against the run's measured noise, and times the CUDA paths when they are built.
- Type stubs for the compiled extension (`_fastdist.pyi`), with a CI check that keeps them in step with the
  bindings.
- A PyPI release workflow using Trusted Publishing, with a TestPyPI dry-run option.
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

- **Performance.** The Poisson, binomial and negative binomial CDFs sum their terms by recurrence instead
  of re-deriving each one, and the batch paths hoist parameter validation and loop-invariant terms out of
  their loops. On the reference machine in `BENCHMARKS.md`, `poisson_cdf` over 100k values went from
  43.5 ms to 1.3 ms (from 0.15x SciPy's speed to 4.8x), `normal_logpdf` got 80% faster, and the uniform and
  normal CDF paths got 22–29% faster.
- `Utils.sigmoid` is scalar-only and raises a `TypeError` naming `Utils.sigmoid_cpu` when given a sequence.
  Its annotation previously advertised sequences, which never worked.
- Annotations use `typing.SupportsFloat` rather than `numbers.Real`, which mypy cannot check, so correct
  code such as `Normal(0.0, 1.0)` no longer reports errors. The package now type-checks cleanly.
- The `exponential` and `poisson` bindings name their rate keyword `lambda_`. The old name, `lambda`, is a
  Python keyword and could never be passed by name.
- The sampling tests are seeded and deterministic, with tolerances at about 5× the estimator's standard
  error, and CI no longer retries failed C++ tests.
- `requirements.txt` is now `requirements-dev.txt`. Runtime dependencies are declared only in the package
  metadata.
- Package metadata moved from `setup.py` into `pyproject.toml`.
- `NDEBUG` is undefined for the `fastdist_tests` target, so its `assert()`-based checks stay live in Release
  builds. They were previously compiled away, meaning the suite reported success without testing anything.
- Repository layout: bindings moved from `python/bindings` to `src/bindings`; tests split into `tests/cpp`
  and `tests/python`; the showcase notebook moved to `examples/`.
- `python_requires` lowered to `>=3.10` and reconciled with the CMake Python requirement, which previously
  disagreed with it.
- `cmake` and `ninja` are declared as build dependencies, so an install from source no longer requires
  CMake to be installed beforehand.

### Known issues

- The name `fastdist` belongs to an unrelated project on PyPI, so this package cannot be published under
  it. The release workflow refuses to upload until the distribution is renamed.
- Sampling draws one variate per call, which makes bulk generation 30–100x slower than numpy. There is no
  batch sampling entry point yet.
- The gamma CDF's iteration ceiling covers shape parameters up to roughly 20000. Beyond that the result
  degrades without warning.
- The `*_cpu` bindings do not expose the `step_size` default that the C++ headers declare, and `step_size`
  is a `double` for the continuous distributions but an `int` for the discrete ones.
- CI compiles the CUDA backend (on Linux, in the type-stub job) but has no GPU runner, so the CUDA
  kernels are not exercised there.

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
