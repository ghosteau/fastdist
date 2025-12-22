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

## Building Wheels for Multiple Python Versions (3.12–3.14)

To generate wheels for all currently supported Python versions:

1. Install each required Python version:
    - https://www.python.org/downloads/

2. From the project root, run:

```powershell
.\build_all.ps1
```

3. Optional cleanup:
    - Add the `-Clean` flag to remove temporary build artifacts and virtual environments after each build:

```powershell
.\build_all.ps1 -Clean
```

When cleanup is enabled, only the final wheel files are retained.

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

- Add moment-generating functions (MGFs)
- Add more robust test coverage
- Add CUDA support
- Add vectorized APIs
- Implement common statistical identities (e.g., Binomial Theorem, Bayes’ Theorem)
- Add utility functions (factorial, gamma, etc.) where not directly supported by the C standard math library
- Add random sampling (RNG) functions for each distribution
- Add additional distributions:
    - Hypergeometric
    - Multinomial
    - Negative binomial
    - Binomial
    - Others
- Add additional log-PDF scalars
- Add comprehensive function documentation

---

## Zach’s TODO

- Add OOP support via Python class wrappers

---

## Contributors

Special thanks to:

- Manny McGrail
- Zach Pipes
