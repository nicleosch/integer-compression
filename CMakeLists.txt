#######################
# Integer Compression #
#######################

# PROJECT
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.15)
project(IntegerCompression VERSION 0.0.1)

# C++ BUILD OPTIONS
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Paths
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)

# Components
# ---------------------------------------------------------------------------
include("${CMAKE_SOURCE_DIR}/src/local.cmake")

# Executable
# ---------------------------------------------------------------------------
add_executable(ic main.cpp)

