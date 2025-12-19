cmake_minimum_required(VERSION 3.20)
project(fastdist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# fastdist library
# -----------------------------
add_library(fastdist_core STATIC
        src/api/version.cpp
        src/api/normal.cpp
        src/api/exponential.cpp
        src/api/poisson.cpp
        src/api/bernoulli.cpp

        src/math/normal.cpp
        src/math/exponential.cpp
        src/math/poisson.cpp
        src/math/bernoulli.cpp
)

target_include_directories(fastdist_core
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# -----------------------------
# pybind11
# -----------------------------
add_subdirectory(python/pybind11)
pybind11_add_module(fastdist
        python/bindings/bindings.cpp
        python/bindings/bernoulli.cpp
        python/bindings/normal.cpp
        python/bindings/poisson.cpp
        python/bindings/exponential.cpp
)

target_link_libraries(fastdist PRIVATE fastdist_core)

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
