cmake_minimum_required(VERSION 3.20)
project(fastdist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# fastdist library
# -----------------------------
add_library(fastdist SHARED
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

target_include_directories(fastdist
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# -----------------------------
# Tests
# -----------------------------
add_executable(fastdist_tests
        tests/basic_tests.cpp
)

target_link_libraries(fastdist_tests
        PRIVATE fastdist
)

target_include_directories(fastdist_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/include
)
