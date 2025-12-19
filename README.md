**General Information:**

- TBD.
- Note that most derivations of things such as means, standard deviations, MGFs, PDFs, and CDFs can be found online.

**Environment Setup and Notes:**
- Make sure to have pybind11 cloned under the python directory.

when cloning the repo: `\
git clone --recurse-submodule https://github.com/ghosteau/fastdist.git \
git submodule update --init --recursive`

To build and compile fastdist:
- Build the project in C++
- This will produce the .pyd file under cmake-build-debug
- Run `python3 python/setup.py bdist_wheel`
    - NOTE: This is currently ONLY working with Python 3.14. Check for more updates in the future as we make the library more accesible
    - This will create the wheel file
- To add to the test.py project open the python venv (3.14) with `.\.venv\Scripts\activate`
- Run `pip install .\dist\fastdist-0.0.1-cp314-cp314-win_amd64.whl --force-reinstall`

- If you want to run the pre-commit (clang-format) locally, you must have Python in your PATH and pre-commit installed via pip on your PC, where then you can run from the project root: `pip install pre-commit`
- Once this is installed, you can run the following command locally: `pre-commit run --all-files` which will format the code on your local machine
    - NOTE: Our repository REQUIRES that you run this before any commits or pull requests because of the value in consistency and readability 

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
- Figure out how to create different module version for each python version
- Add OOP support via Python class wrapper
- ~~Add pybind11 as a github submodule~~
