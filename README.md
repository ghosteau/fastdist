# fastdist

## General Information

- This library provides high-performance implementations of common probability distributions and related statistical
  functions.
- Derivations of means, variances, moment-generating functions (MGFs), PDFs, and CDFs are widely available online and
  are therefore not duplicated here.

---

## Environment Setup and Build Notes

### Cloning the Repository

This project uses Git submodules. Clone the repository recursively:

```bash
git clone --recurse-submodule https://github.com/ghosteau/fastdist.git
git submodule update --init --recursive
```

---

## Building and Installing `fastdist`

### Building the C++ Extension and Python Wheel

1. Build the C++ project using CMake.
    - This produces the compiled extension (`.pyd` on Windows) in your CMake build directory (e.g.,
      `cmake-build-debug`).

2. From the **project root**, build the Python wheel:

```bash
python3 setup.py bdist_wheel
```

**Important:**

- This command must be run from the project root.

3. Install the generated wheel:

```bash
pip install .\dist\fastdist-<version>-cpXXX-cpXXX-win_amd64.whl --force-reinstall
```

---

## Creating A Distribution Class

### Writing the C++ Code (Unfinished)

1. Create functions (cpp and h)
2. Add api/src files to CMakeLists.txt

### Creating the Python Bindings

1. Under python/bindings create a new file named `<distribution_name>.cpp`.
2. Add the following code to `python/bindings/<distribution_name>.cpp`:

```
namespace py = pybind11;

void bind_<distribution_nam>(py::module_ &m) {
    m.def("example_function", &fastdist::math::example_function,
    py::arg("example_var1"), py::arg("example_var2"), R"pbdoc(Example function documentation.)pbdoc");
}
```

3. Add the following lines of code to `python/bindings/bindings.cpp` (in alphabetical order):
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
`python/bindings/normal.cpp` contains the pybind11 bindings. \
`python/bindings/bindings.cpp` contains the module bindings. \
`python/fastdist/distributions/normal.py` contains the Python class definition.

---

## Updating the Version
Only update the version from the CMakeLists.txt at the line:
`project(fastdist VERSION x.y.z LANGUAGES CXX)`

## Building Wheels for Multiple Python Versions (3.12–3.14)

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

- Pybind11 integrated as a submodule for modular C++/Python bindings
- Full Python support for all currently supported builds

---

Long-term plans:

- Add Hypergeometric Distribution
- Make auto_tune() dynamically find the sign flip
- Add CUDA/Batch extern functions
- Set up CI for cuda tests
- Add cuda implementation for all classes
- Fix up the python-distro.yml file to be more efficient and comprehensive
- Look into the usage of @classmethod and check for redundancies in the Python classes
- Add specific parameters in all return _core.<class>_<func>(x, a, b) → (x=x, a=a, b=b)
- Check for all isfinite values (currently only set up in normal)
- Merge validation checks and CUDA availability into a singular function for cleanliness
- Use size_t instead of int in all cuda files
- Add memory constraint option to cuda where if you have limited gpu memory you can set what your limit for streaming is
- Update all docstrings to match each other and be comprehensive
- Create new cuda tests
- Refine Utils class to be more efficient and comprehensive
- Add batch and cuda functions to the C API
- Update pynvml to nvidia-ml-py
- Precalculate reused values on CPU and send to GPU
- Make a full, comprehensive documentation page
- In the future, try to get the library on pip 
---

## Contributors

Special thanks to:

- Manny McGrail
- Zach Pipes
