cmake_minimum_required(VERSION 3.20)
project(fastdist VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Generated version header
# -----------------------------
configure_file(
        ${PROJECT_SOURCE_DIR}/include/fastdist/version.h.in
        ${PROJECT_BINARY_DIR}/generated/fastdist/version.h
        @ONLY
)
message(STATUS "fastdist version ${PROJECT_VERSION}")

# -----------------------------
# CUDA Backend
# -----------------------------
option(FASTDIST_ENABLE_CUDA "Enable CUDA backend" OFF)

if (FASTDIST_ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA backend enabled")
else ()
    message(STATUS "CUDA backend disabled")
endif ()

# -----------------------------
# python
# -----------------------------
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Building against Python ${Python_VERSION} at ${Python_EXECUTABLE}")

# -----------------------------
# pybind11
# -----------------------------
set(PYBIND11_FINDPYTHON ON)

# Use the pybind11 installed in the active environment (already declared in
# pyproject.toml build-system.requires) so headers always match the interpreter.
if (NOT pybind11_DIR)
    execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmakedir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _pybind11_probe_result
            ERROR_QUIET
    )
    if (_pybind11_probe_result EQUAL 0)
        set(pybind11_DIR "${_pybind11_cmakedir}")
    else ()
        # No pip pybind11 in this interpreter; fall back to a system install
        # (e.g. Debian's pybind11-dev), which ships its own CMake config.
        message(STATUS "pip pybind11 not found; falling back to a system installation")
    endif ()
endif ()

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Using pybind11 ${pybind11_VERSION} from ${pybind11_DIR}")

pybind11_add_module(_fastdist
        python/bindings/bindings.cpp
        python/bindings/bernoulli.cpp
        python/bindings/normal.cpp
        python/bindings/poisson.cpp
        python/bindings/exponential.cpp
        python/bindings/binomial.cpp
        python/bindings/discrete_uniform.cpp
        python/bindings/uniform.cpp
        python/bindings/geometric.cpp
        python/bindings/negative_binomial.cpp
        python/bindings/gamma.cpp
        python/bindings/beta.cpp
        python/bindings/chi_square.cpp
        python/bindings/utils.cpp

        src/wrappers/normal_wrapper.cpp
        src/wrappers/poisson_wrapper.cpp
        src/wrappers/bernoulli_wrapper.cpp
        src/wrappers/exponential_wrapper.cpp
        src/wrappers/uniform_wrapper.cpp
        src/wrappers/utils_wrapper.cpp
)

# -----------------------------
# fastdist library
# -----------------------------
add_library(fastdist_core STATIC
        src/api/version.cpp
        src/api/normal.cpp
        src/api/exponential.cpp
        src/api/poisson.cpp
        src/api/bernoulli.cpp
        src/api/binomial.cpp
        src/api/discrete_uniform.cpp
        src/api/uniform.cpp
        src/api/geometric.cpp
        src/api/negative_binomial.cpp
        src/api/gamma.cpp
        src/api/beta.cpp
        src/api/chi_square.cpp
        src/api/utils.cpp

        src/math/normal.cpp
        src/math/exponential.cpp
        src/math/poisson.cpp
        src/math/bernoulli.cpp
        src/math/binomial.cpp
        src/math/discrete_uniform.cpp
        src/math/uniform.cpp
        src/math/geometric.cpp
        src/math/negative_binomial.cpp
        src/math/gamma.cpp
        src/math/beta.cpp
        src/math/chi_square.cpp
        src/math/utils.cpp
)

# -----------------------------
# CUDA Sources
# -----------------------------
if (FASTDIST_ENABLE_CUDA)
    target_compile_definitions(fastdist_core
            PUBLIC FASTDIST_ENABLE_CUDA
    )
    target_sources(fastdist_core PRIVATE
            src/cuda/normal/pdf.cu
            src/cuda/normal/logpdf.cu
            src/cuda/normal/cdf.cu
            src/cuda/normal/mgf.cu
            src/cuda/normal/cgf.cu

            src/cuda/poisson/cdf.cu
            src/cuda/poisson/cgf.cu
            src/cuda/poisson/mgf.cu
            src/cuda/poisson/pmf.cu

            src/cuda/bernoulli/cdf.cu
            src/cuda/bernoulli/cgf.cu
            src/cuda/bernoulli/mgf.cu
            src/cuda/bernoulli/pmf.cu

            src/cuda/exponential/cdf.cu
            src/cuda/exponential/cgf.cu
            src/cuda/exponential/mgf.cu
            src/cuda/exponential/pdf.cu

            src/cuda/uniform/cdf.cu
            src/cuda/uniform/cgf.cu
            src/cuda/uniform/mgf.cu
            src/cuda/uniform/pdf.cu

            src/cuda/utils/sigmoid.cu
            src/cuda/utils/logit.cu
            src/cuda/utils/euclidean_distance.cu
            src/cuda/utils/manhattan_distance.cu
            src/cuda/utils/cosine_similarity.cu
    )
    target_sources(_fastdist PRIVATE
            src/cuda/normal/pdf.cu
            src/cuda/normal/logpdf.cu
            src/cuda/normal/cdf.cu
            src/cuda/normal/mgf.cu
            src/cuda/normal/cgf.cu

            src/cuda/poisson/cdf.cu
            src/cuda/poisson/cgf.cu
            src/cuda/poisson/mgf.cu
            src/cuda/poisson/pmf.cu

            src/cuda/bernoulli/cdf.cu
            src/cuda/bernoulli/cgf.cu
            src/cuda/bernoulli/mgf.cu
            src/cuda/bernoulli/pmf.cu

            src/cuda/exponential/cdf.cu
            src/cuda/exponential/cgf.cu
            src/cuda/exponential/mgf.cu
            src/cuda/exponential/pdf.cu

            src/cuda/uniform/cdf.cu
            src/cuda/uniform/cgf.cu
            src/cuda/uniform/mgf.cu
            src/cuda/uniform/pdf.cu

            src/cuda/utils/sigmoid.cu
            src/cuda/utils/logit.cu
            src/cuda/utils/euclidean_distance.cu
            src/cuda/utils/manhattan_distance.cu
            src/cuda/utils/cosine_similarity.cu
    )
    set_target_properties(fastdist_core PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # CUDA Runtime Linking
    find_package(CUDAToolkit REQUIRED) # Finds Toolkit
    target_link_libraries(fastdist_core PUBLIC CUDA::cudart) # Links Runtime library

    target_include_directories(fastdist_core PRIVATE src)
    target_include_directories(_fastdist PRIVATE src)
endif ()

set_target_properties(fastdist_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(fastdist_core
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/fastdist>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
        ${CUDAToolkit_INCLUDE_DIRS}
        $<INSTALL_INTERFACE:include>
)

# Linking pybind11 to Python module
target_link_libraries(_fastdist PRIVATE pybind11::module)

# Connecting fastdist(python) and fastdist_core(cxx) libraries
target_link_libraries(_fastdist PRIVATE fastdist_core)

# -----------------------------
# Tests
# -----------------------------
add_executable(fastdist_tests
        tests/test_bernoulli.cpp
        tests/test_binomial.cpp
        tests/test_negative_binomial.cpp
        tests/test_discrete_uniform.cpp
        tests/test_exponential.cpp
        tests/test_geometric.cpp
        tests/test_normal.cpp
        tests/test_poisson.cpp
        tests/test_uniform.cpp
        tests/test_gamma.cpp
        tests/test_beta.cpp
        tests/test_chi_square.cpp
        tests/test_utils.cpp

        tests/test_app.cpp
)

target_link_libraries(fastdist_tests
        PRIVATE fastdist_core
)

target_include_directories(fastdist_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/include
)

# Link CUDA to Python module if enabled
if (FASTDIST_ENABLE_CUDA)
    target_link_libraries(_fastdist PRIVATE CUDA::cudart)
endif ()