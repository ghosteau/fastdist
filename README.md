**General Information:**

- TBD.
- Note that most derivations of things such as means, standard deviations, MGFs, PDFs, and CDFs can be found online.

**Environment Setup and Notes:**
When cloning the repo: \
` git clone --recurse-submodule https://github.com/ghosteau/fastdist.git` \
` git submodule update --init --recursive`

To build and compile fastdist:

- Build the project in C++
- This will produce the .pyd file under cmake-build-debug
- Run `python3 python/setup.py bdist_wheel` from the root directory (very important)
    - NOTE: This is currently ONLY working with Python 3.14. Check for more updates in the future as we make the library
      more accessible
    - This will create the wheel file
- Run `pip install .\dist\fastdist-0.0.1-cpXXX-cpXXX-win_amd64.whl --force-reinstall`

To create all current (3.12-3.14) python versions:

- You will have to have the specific python version installed to create the wheel
    - https://www.python.org/downloads/
- From root run `.\build_all.ps1`
    - You can add -Clean to the command to clean up all build directories and only produce the wheel files.

- If you want to run the pre-commit (clang-format) locally, you must have Python in your PATH and pre-commit installed
  via pip on your PC, where then you can run from the project root: `pip install pre-commit`
- Once this is installed, you can run the following command locally: `pre-commit run --all-files` which will format the
  code on your local machine
    - NOTE: Our repository REQUIRES that you run this before any commits or pull requests because of the value in
      consistency and readability

**Release Notes**

- TBD.

**TODO:**

- Add MGFs
- Add more robust testing
- Add CUDA support
- Add vectorized APIs
- Binomial theorem, Bayes' Theorem, and other statistically relevant identities and formulas
- Add utility functions such as factorial, gamma, and others, where and if not directly supported in the C standard math
  library
- Add sampling functions for each distribution (RNG)
- Add more distributions such as hypergeometric, multinomial, negative binomial, binomial, etc...
- Add other log PDF scalars
- Add function docs

**Zach's TODO:**

- ~~Add formatter (like Black in Python) ⇒ use clang-format~~
- ~~Figure out how to create different module version for each python version~~
- Add OOP support via Python class wrapper
- ~~Add pybind11 as a GitHub submodule~~
- ~~Test cmake and build in GitHub actions~~
- ~~Add Python unit tests and make action for it~~

Special thanks to our contributors:

- Manny McGrail
- Zach Pipes
