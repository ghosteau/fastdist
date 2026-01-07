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
python3 python/setup.py bdist_wheel
```

**Important:**

- This command must be run from the project root.

3. Install the generated wheel:

```bash
pip install .\dist\fastdist-0.0.1-cpXXX-cpXXX-win_amd64.whl --force-reinstall
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

- TBD

---

## Manny's TODO

- Add moment-generating functions (MGFs) and CGFs
- Add more robust test coverage
- Add random sampling (RNG) functions for each distribution
- Add additional log-PDF scalar functions
- Add comprehensive function documentation
- Add docstrings / comment block documentation to functions in C++ and Python
- Make comments within math directory a bit more consistent

---

## Zach’s TODO

Pre-release:

- Add OOP Python support and tests
- Add vectorized APIs
- Set up automated cuda optimization (STREAMING_THRESHOLD and N for each function)
- Add cuda implementation for pre-release classes:
    - Normal
    - Poisson
    - Exponential
    - Bernoulli
    - Continuous Uniform
    - Utils
        - Euclidean Distance
        - Manhattan Distance
        - Cosine Similarity
        - Sigmoid (vectorized)
        - Logit (vectorized)
- Check for using more than your GPUs memory in cuda functions
- Create new cuda tests
- Test cuda error catching
- "Overload" python functions for deciding whether to use scalar, batch, or cuda functions

  Long-term plans:

- Fix -enableCuda not working in GitHub Actions
- Add CUDA/Batch extern functions
- Set up CI for cuda tests
- Create benchmarks for performance comparisons
- Add cuda implementation for all classes
- Fix up the python-distro.yml file to be more efficient and comprehensive
- Look into the usage of @classmethod and check for redundancy
- Update instances of "k: int | float" to "k: Union[int, float]" to support python pre 3.10
- Use numbers.Real for type checking? (includes Decimal and Fraction types) ((from typing library))
- Add specific parameters in all return _core.<class>_<func>(x, a, b) → (x=x, a=a, b=b)
- Check for all isfinite values (currently only set up in normal)
- Merge validation checks into a separate function for cleanliness
- Use size_t instead of int in all cuda files

---

## Contributors

Special thanks to:

- Manny McGrail
- Zach Pipes
