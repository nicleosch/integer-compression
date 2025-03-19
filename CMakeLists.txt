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

# COMPILE OPTIONS
# ---------------------------------------------------------------------------
add_compile_options(-mavx2 -mavx512f)

# PATHS
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)


# DEPENDENCIES
# ---------------------------------------------------------------------------
set(VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor/")

include("${VENDOR_DIR}/SIMDcomp/simdcomp.cmake")
include_directories(${SIMDCOMP_INCLUDE_DIR})

# COMPONENTS
# ---------------------------------------------------------------------------
include("${CMAKE_SOURCE_DIR}/src/local.cmake")

add_executable(ic main.cpp)
target_link_libraries(ic ${PROJECT_NAME} SIMDcomp)