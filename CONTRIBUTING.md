# Contributing to fastdist

## Project layout

```
include/fastdist/    public C++ headers (math/, cuda/, wrappers/)
src/math/            C++ implementations
src/api/             C API surface
src/cuda/            CUDA kernels (.cu)
src/wrappers/        wrappers bridging the C++ core to the bindings
src/bindings/        pybind11 bindings
python/fastdist/     the Python package
tests/cpp/           C++ unit tests
tests/python/        pytest suite
examples/            notebooks
```

The C++ core builds as a static library, `fastdist_core`. The Python extension, `_fastdist`, links against
it and is what `python/fastdist` imports.

---

## Building from source

You need CMake 3.20+, a C++20 compiler, and Python 3.10+.

```bash
pip install -r requirements.txt
pip install .
```

To work on the C++ side directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

On Windows the Visual Studio generator is multi-config, so `CMAKE_BUILD_TYPE` is ignored — pick the
configuration at build time instead:

```powershell
cmake -S . -B build
cmake --build build --config Release --parallel
```

CMake selects the newest Visual Studio it can find. Set `CMAKE_GENERATOR` to pin a specific one.

### With CUDA

```bash
FASTDIST_ENABLE_CUDA=1 pip install .
```

or `-DFASTDIST_ENABLE_CUDA=ON` when configuring CMake directly. Requires an NVIDIA GPU and the CUDA
toolkit; on Windows, a Visual Studio installation is required.

---

## Running the tests

### C++

Each test file is registered as its own CTest entry, so failures are isolated and you can run one at a time.

Linux / macOS:

```bash
ctest --test-dir build --output-on-failure          # all 13
ctest --test-dir build -R geometric                 # one (regex)
ctest --test-dir build -j 8                         # in parallel
```

Windows. The Visual Studio generator is multi-config, so **`-C Release` is required** — without it CTest
cannot tell which configuration to run and reports `Test not available without configuration`:

```powershell
ctest --test-dir build -C Release --output-on-failure
ctest --test-dir build -C Release -R geometric
ctest --test-dir build -C Release -j 8
```

Match `-C` to whatever you passed to `cmake --build --config`.

The binary also takes test names directly:

```bash
./build/fastdist_tests                              # all, in declaration order
./build/fastdist_tests geometric uniform            # just those two
```

```powershell
.\build\Release\fastdist_tests.exe                  # note the config subdirectory
.\build\Release\fastdist_tests.exe geometric uniform
```

Note the suite is built on `assert()`. `CMakeLists.txt` undefines `NDEBUG` for the test target specifically,
so assertions stay live in Release builds — without that, `ctest` would pass while checking nothing.

Three assertions in the RNG blocks are known-flaky; CI runs `ctest --repeat until-pass:3` to absorb them.

### Python

```bash
pytest                       # testpaths in pyproject.toml points at tests/python
pytest tests/python -q
```

**The tests import the installed package, not `python/fastdist`.** Editing Python or C++ source changes
nothing the tests see until you reinstall:

```bash
pip install . --force-reinstall --no-deps
```

Skipping this is an easy way to spend an hour debugging a failure that source already fixed.

---

## Building wheels for several Python versions

`build_all.ps1` (Windows, PowerShell) builds wheels locally:

```powershell
.\build_all.ps1
```

Flags:

| Flag | Effect |
|---|---|
| `-Clean` | remove build artifacts and virtualenvs after each build |
| `-EnableCuda` | build with CUDA support |
| `-PythonVersion 3.13` | build only that version |
| `-PipInstall 3.14` | install the built wheel afterwards |

It defaults to building every supported version, 3.10 through 3.14, and skips any that `py -<version>`
cannot find rather than failing. CI covers the same range across Linux, Windows and macOS via
`cibuildwheel`.

---

## Adding a distribution

Normal is the reference implementation; follow it.

1. **C++** — declare in `include/fastdist/math/<name>.h`, implement in `src/math/<name>.cpp`, and add both
   to the `fastdist_core` source list in `CMakeLists.txt`.
2. **Bindings** — create `src/bindings/<name>.cpp`:

   ```cpp
   namespace py = pybind11;

   void bind_<name>(py::module_ &m) {
       m.def("example_function", &fastdist::math::example_function,
             py::arg("x"), py::arg("a"),
             R"pbdoc(Example function documentation.)pbdoc");
   }
   ```

   Then in `src/bindings/bindings.cpp`, add `void bind_<name>(py::module &m);` and call `bind_<name>(m);`
   inside `PYBIND11_MODULE()`, alphabetically. Add the file to `CMakeLists.txt`.

   Keep `py::arg` names identical to the keywords the Python wrapper passes — pybind11 matches keywords
   strictly, and a mismatch is a `TypeError` at call time rather than a build error.

3. **Python** — add `python/fastdist/distributions/<name>.py`, export the class from
   `distributions/__init__.py` (`__all__`) and from `fastdist/__init__.py`.
4. **Tests** — add `tests/cpp/test_<name>.cpp`, declare it in `tests/cpp/test_app.cpp` (forward
   declaration plus an entry in the `tests` table), add the name to `FASTDIST_TEST_CASES` in
   `CMakeLists.txt`, and add `tests/python/test_<name>.py`.

Statistical assertions should size their tolerance from the estimator's standard error, not a round number.
A tolerance at 2σ fails several percent of runs.

---

## Type stubs

`_fastdist` is a compiled extension, so type checkers cannot introspect it. `python/fastdist/_fastdist.pyi`
declares its API for them. It is **generated** — regenerate it after changing any binding:

```bash
pip install pybind11-stubgen
pybind11-stubgen fastdist._fastdist -o stubs
cp stubs/fastdist/_fastdist.pyi python/fastdist/_fastdist.pyi
```

Keep the header comment at the top when you replace it.

Two things to know:

- The stub is generated from a **CPU-only** build, matching the published wheels. `*_cuda` bindings live
  inside `#ifdef FASTDIST_ENABLE_CUDA` and are absent from a CPU build, so they are absent from the stub.

---

## Versioning

`CMakeLists.txt` is the single source of truth:

```cmake
project(fastdist VERSION x.y.z LANGUAGES CXX)
```

`setup.py` parses it for the package metadata and the C++ header is generated from it, so nothing else
needs editing. `tests/python/test_version.py` asserts the three agree.

---

## Formatting

`clang-format` is enforced on all C/C++/CUDA sources via pre-commit, and checked in CI.

```bash
pip install pre-commit
pre-commit run --all-files
```

Run it before opening a pull request.

---

## CI

| Workflow | Trigger | What it does                                                                                              |
|---|---|-----------------------------------------------------------------------------------------------------------|
| `clang-format.yml` | push to main/develop, all PRs | formatting check                                                                                          |
| `python-distro.yml` | push to main/develop, all PRs | C++ build, `ctest`, wheel build, pytest                                                                   |
| `wheels.yml` | push to main | `cibuildwheel` across Linux/Windows/macOS for 3.10–3.14, plus sdist build and an install-from-sdist check |
