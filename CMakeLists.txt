cmake_minimum_required(VERSION 3.20)
project(fastdist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development)

# -----------------------------
# pybind11
# -----------------------------
# Setting up python-pybind11 connection
set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYBIND11_PYTHON_VERSION ${Python_VERSION_STRING})

# Initializing pybind11 library
include(FetchContent)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.12.1
)

FetchContent_MakeAvailable(pybind11)

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
    )
    target_sources(_fastdist PRIVATE
            src/cuda/normal/pdf.cu
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
endif ()

set_target_properties(fastdist_core PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(fastdist_core
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/fastdist/cuda>
)

# Linking pybind11 to Python module
target_link_libraries(_fastdist PRIVATE pybind11::module)

# Connecting fastdist(python) and fastdist_core(cxx) libraries
target_link_libraries(_fastdist PRIVATE fastdist_core)

# -----------------------------
# Tests
# -----------------------------
add_executable(fastdist_tests
        tests/basic_tests.cpp
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