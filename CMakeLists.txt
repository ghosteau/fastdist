cmake_minimum_required(VERSION 3.20)
project(fastdist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(fastdist SHARED
        src/api/version.cpp
)

target_include_directories(fastdist
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)