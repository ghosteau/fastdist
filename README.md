# fastdist

## General Information

- This library provides high-performance implementations of common probability distributions and related statistical
  functions.
- Derivations of means, variances, moment-generating functions (MGFs), PDFs, and CDFs are widely available online and
  are therefore not duplicated here.

---

## Environment Setup and Build Notes

### Cloning the Repository

```bash
git clone https://github.com/ghosteau/fastdist.git
```

---

## Building and Installing `fastdist`

### Installing

From the **project root**:

```bash
pip install .
```

That is the whole thing. `setup.py` drives CMake, and `pyproject.toml` declares
the build dependencies (including `pybind11` and `cmake`), so pip provisions
them in an isolated build environment.

To build a wheel without installing it:

```bash
pip install build
python -m build --wheel
```

The wheel lands in `dist/`. (`python setup.py bdist_wheel` still works but is
deprecated upstream; prefer `python -m build`.)

### Building the C++ project directly

Needed when working on the C++ side, running the C++ tests, or using an IDE's
CMake integration:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target fastdist_tests --parallel
ctest --test-dir build --output-on-failure
```

On Windows the Visual Studio generator is multi-config, so `CMAKE_BUILD_TYPE`
is ignored -- pick the configuration at build and test time instead:

```powershell
cmake -S . -B build
cmake --build build --target fastdist_tests --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

**If CMake reports it cannot find pybind11**, it is resolving pybind11 from the
active interpreter and that interpreter has none installed. Either install it
(`pip install pybind11`) or point CMake at one explicitly:

```bash
cmake -S . -B build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
```

Pass `-DPython_EXECUTABLE=...` as well if the interpreter you want is not the
first one on `PATH`.

### Building with CUDA

```bash
cmake -S . -B build -DFASTDIST_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

`Normal.is_cuda_available()` reports whether the extension was built with the
backend; the Python classes dispatch to it automatically above the per-function
thresholds in `fastdist.config`.

**The CUDA toolkit and the MSVC toolset have to be compatible versions.** This
is the failure most likely to stop you, and it does not announce itself
clearly: `nvcc` aborts with

```
nvcc error : 'cudafe++' died with status 0xC0000409
```

on every `.cu` file. That is `cudafe++` crashing on standard library headers
from an MSVC newer than the toolkit supports. `-allow-unsupported-compiler`
silences the version *check* but does not fix the incompatibility.

Each CUDA release supports host compilers up to a specific MSVC version --
CUDA 12.4 tops out at MSVC 19.39 (Visual Studio 17.9), so MSVC 19.42
(VS 17.12) fails. Check with `nvcc --version` and `cl` (in a developer prompt),
then either:

- upgrade the CUDA toolkit to one that supports your MSVC, or
- install the older MSVC toolset alongside the current one through the Visual
  Studio Installer (Individual components -> "MSVC v143 ... build tools
  (v14.39)") and point CMake at it with `-T version=14.39`.

Note also that the CUDA toolkit installs its Visual Studio MSBuild integration
into whichever VS instance it finds. If you have both Build Tools and a full
Visual Studio install, the integration may land in the one without the C++
workload, and the Visual Studio generator will then report "No CUDA toolset
found" even though `nvcc` is on `PATH`. The Ninja generator does not use that
integration at all:

```bash
cmake -S . -B build -G Ninja -DFASTDIST_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

Ninja needs the MSVC environment on `PATH`, so run it from a "x64 Native Tools
Command Prompt" (or after `vcvars64.bat`).

---

## Creating A Distribution Class

### Writing the C++ Code (Unfinished)

1. Create functions (cpp and h)
2. Add api/src files to CMakeLists.txt

### Creating the Python Bindings

1. Under src/bindings create a new file named `<distribution_name>.cpp`.
2. Add the following code to `src/bindings/<distribution_name>.cpp`:

```
namespace py = pybind11;

void bind_<distribution_nam>(py::module_ &m) {
    m.def("example_function", &fastdist::math::example_function,
    py::arg("example_var1"), py::arg("example_var2"), R"pbdoc(Example function documentation.)pbdoc");
}
```

3. Add the following lines of code to `src/bindings/bindings.cpp` (in alphabetical order):
    1. `void bind_<distribution_name>(py::module &m);`
    2. `bind_<distribution_name>(m);` under `PYBIND11_MODULE()`

### Creating Python Classes

1. In `python/fastdist/__init__.py` add the import statement:
    - `from .distributions import <ClassName>`
2. Also in `python/fastdist/distributions/__init__.py` add the class to the `__all__` variable:
    - `__all__ = ["Normal", <ClassName>, ... ]`
3. Create a new file in `python/fastdist/distributions/<distribution_name>.py` and add the class definitions.

### Notes

The normal distribution is a good reference for creating new distributions. \
`src/api/normal.h` contains the C++ function declarations. \
`src/math/normal.cpp` contains the C++ function definitions. \
`src/bindings/normal.cpp` contains the pybind11 bindings. \
`src/bindings/bindings.cpp` contains the module bindings. \
`python/fastdist/distributions/normal.py` contains the Python class definition.

---

## Updating the Version
Only update the version from the CMakeLists.txt at the line:
`project(fastdist VERSION x.y.z LANGUAGES CXX)`

## Building Wheels for Multiple Python Versions (3.10–3.14)

To generate wheels for all currently supported Python versions:

1. Install each required Python version:
    - https://www.python.org/downloads/

2. Install CMake (using CLion's CMake DOES NOT WORK, you need it installed on your system):
    - https://cmake.org/download/

3. From the project root, run:

```powershell
.\build_all.ps1
```

4. Optional Flags:
    - `-Clean`: Removes temporary build artifacts and virtual environments after each build:
    - **`-enableCuda`: Enables CUDA support if a compatible NVIDIA GPU and CUDA toolkit are available.
    - `-PythonVersion`: Specifies a particular Python version to build for (e.g., `-PythonVersion 3.12`).
        - Ex: `-PythonVersion 3.13` will only build the wheel for Python 3.13.
        - Only Python 3.12, 3.13, 3.14 are supported.
    - `-PipInstall`: Installs the built wheel using the specified version after building.
        - Ex: `-PipInstall 3.14` will install the wheel for Python 3.14.

When cleanup is enabled, only the final wheel files will remain.

** If you are trying to build with CUDA enabled, it is REQUIRED that you have Visual Studio 2022 (version 17) installed.

---

## Reproducible Sampling

Every `*_sample()` function draws from one shared Mersenne Twister engine.
Seeding it makes a run reproducible:

```python
import fastdist

fastdist.seed(12345)
a = [fastdist.Normal(0, 1).sample() for _ in range(5)]

fastdist.seed(12345)
b = [fastdist.Normal(0, 1).sample() for _ in range(5)]

assert a == b            # exactly equal, not merely close

fastdist.seed_from_entropy()   # back to non-deterministic
```

From C++, the same thing lives in `fastdist/math/rng.h`:

```cpp
#include <fastdist/math/rng.h>

fastdist::math::seed_rng(12345);
fastdist::math::seed_rng_from_entropy();
```

Two limits are worth knowing before relying on this:

1. **Seeding is per-thread.** The engine is `thread_local`, so a worker thread
   that has not been seeded keeps its own entropy-initialised stream. This is
   what makes concurrent sampling lock-free; it also means one `seed()` call
   does not cover threads you spawn.

2. **Reproducible per platform, not across them.** `std::mt19937` is specified
   bit-for-bit by the C++ standard, but the distribution adaptors built on it
   (`std::normal_distribution` and friends) are not. The same seed therefore
   produces different samples under libstdc++, libc++ and MSVC. A seed pins a
   run on one platform and toolchain, not across all of them.

---

## Benchmarks

Performance is measured against SciPy and recorded in
[BENCHMARKS.md](BENCHMARKS.md), with the raw JSON for every run kept under
`benchmarks/results/`.

```bash
pip install scipy          # baseline only; not needed to build or use fastdist
python benchmarks/run.py                  # full suite, writes a report
python benchmarks/run.py --quick          # one array size, for a fast check
python benchmarks/compare.py --latest     # diff the two newest reports
```

When the extension is built with `FASTDIST_ENABLE_CUDA=ON`, the suite also
times the `*_cuda` entry points against this library's own `*_cpu` path, so the
reported speedup is the one a caller actually decides on. GPU timings include
the host-to-device copy and the copy back, since a caller cannot avoid those.
The CUDA cases are skipped entirely on a CPU-only build.

`compare.py` exits non-zero if any case regressed by more than 5%, so it can
gate a change. Every case checks that fastdist and the baseline agree
numerically before either is timed -- a speedup on a wrong answer is not a
speedup.

Add an entry to BENCHMARKS.md when a release ships, or when a change is made
specifically to move performance. Generate the table with
`python benchmarks/table.py --latest` rather than transcribing numbers.

---

## Testing

```bash
ctest --test-dir build --output-on-failure   # C++
pytest                                       # Python
```

### Tolerances in sampling tests

The RNG test blocks draw a large sample and compare its mean and variance
against theory. Two rules keep those honest:

1. **Seed first.** Every RNG block calls `seed_rng()` before sampling, so it is
   deterministic: it either always passes or always fails on a given toolchain,
   never intermittently.

2. **Size the tolerance from the estimator's standard error**, not from a round
   number. Each block's tolerance is roughly 5x the standard error of the
   statistic being checked, and the SE is recorded in a comment at the site.

Round-number tolerances are how this suite acquired a 7.7% flake rate: several
sat near 2 sigma of their estimator's own noise and failed at about the rate a
2 sigma bound fails (ghosteau/fastdist#2). If you change `N` or a distribution
parameter, recompute the standard error and resize the tolerance with it.

---

## Code Formatting and Pre-Commit Hooks

This repository enforces consistent formatting using `clang-format`.

To run formatting locally:

1. Ensure Python is available in your `PATH`.
2. Install `pre-commit`:

```bash
pip install pre-commit
```

3. Run formatting checks from the project root:

```bash
pre-commit run --all-files
```

**Note:**  
Running pre-commit is **required** before submitting commits or pull requests to ensure consistency and readability
across the codebase.

---

## Release Notes

### v0.1.0 — Initial Pre-Release

This is the first public pre-release of fastdist, establishing the core architecture, API surface, and build system.
This release focuses on correctness, performance, and extensibility across C++, Python, and CUDA backends.

Distributions:

- Bernoulli
- Beta
- Binomial
- Chi-square
- Discrete Uniform
- Exponential
- Gamma
- Geometric
- Negative Binomial
- Normal
- Poisson
- Uniform

All distributions include PDF and CDF implementations.
Moment-generating functions (MGFs) are available for a subset of distributions and may be expanded or modified in the
future.

Statistical Utilities:

- Chebyshev’s inequality
- Bayes’ theorem
- Law of Total Probability
- Sigmoid and logit functions
- Euclidean distance
- Manhattan distance
- Cosine similarity
- Coefficient of variation
- Covariance
- Combinatorial utilities (choose, factorial, binomial theorem)
- Special functions (gamma and log-gamma)

CUDA Functionality (Early Support Stage):

- Accelerated numerical computation on compatible NVIDIA GPUs
- GPU-backed random number generation (RNG)
- Clustered and batched computation workflows
- Compatible Classes:
    - Normal
    - Uniform
    - Exponential
    - Bernoulli
    - Utils functions:
        - Euclidean distance
        - Manhattan distance
        - Cosine similarity
        - Logit
        - Sigmoid

Testing and CI:

- Initial unit tests covering core functionality and use-cases
- GitHub Actions pipelines ensure correctness across updates and patches

Python Bindings:

- Pybind11 resolved from the build environment for modular C++/Python bindings
- Full Python support for all currently supported builds

---

Long-term plans:

Performance (see [BENCHMARKS.md](BENCHMARKS.md) for what is measured today):

- Add batch sampling entry points (`normal_sample_batch(n)` returning an array).
  Sampling is currently 30-100x slower than numpy because every variate crosses
  the Python/C++ boundary individually. This is the single largest gap in the
  library.
- Speed up `poisson_pmf`, still ~0.8x SciPy. The per-element `lgamma` dominates
  and is not loop-invariant, so it needs a different evaluation strategy.
- Extend batch invariant hoisting to the distributions the benchmark suite does
  not yet cover (binomial, geometric, beta, gamma, chi-square, discrete uniform,
  negative binomial). The pattern is established in `normal.cpp`.
- Precalculate reused values on CPU and send to GPU
- Add a memory-constraint option to CUDA, so a limited GPU can cap its streaming
  budget

Correctness and API:

- Add Hypergeometric Distribution
- Make auto_tune() dynamically find the sign flip
- Check for all isfinite values (currently only set up in normal)
- Add specific parameters in all return _core.<class>_<func>(x, a, b) → (x=x, a=a, b=b)
- Expose the `step_size` defaults the C++ headers declare through the pybind11
  bindings; callers currently have to pass it explicitly
- Make `step_size` consistently typed -- it is `double` on the continuous batch
  functions and `int` on the discrete ones
- Merge validation checks and CUDA availability into a singular function for cleanliness
- Look into the usage of @classmethod and check for redundancies in the Python classes
- Refine Utils class to be more efficient and comprehensive
- Seed the CUDA RNG alongside the CPU engine, so `seed()` covers both backends

CUDA:

- Add CUDA/Batch extern functions
- Set up CI for cuda tests
- Add cuda implementation for all classes
- Create new cuda tests
- Use size_t instead of int in all cuda files
- Add batch and cuda functions to the C API
- Benchmark the CUDA backend and record it in BENCHMARKS.md

Tooling and docs:

- Fix up the python-distro.yml file to be more efficient and comprehensive
- Update all docstrings to match each other and be comprehensive
- Update pynvml to nvidia-ml-py
- Make a full, comprehensive documentation page
- In the future, try to get the library on pip

Done since v0.1.0:

- ~~Seedable RNG for reproducible sampling~~ (`fastdist.seed()`)
- ~~Flaky RNG tests~~ (tolerances sized from estimator standard error; #2)
- ~~Performance measurement and evidence log~~ (`benchmarks/`, BENCHMARKS.md)

---

## Contributors

Special thanks to:

- Manny McGrail
- Zach Pipes
