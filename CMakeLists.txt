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
include("${CMAKE_SOURCE_DIR}/vendor/GoogleTest/googletest.cmake")

include_directories(
    ${SIMDCOMP_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIR}
)

# COMPONENTS
# ---------------------------------------------------------------------------
include("${CMAKE_SOURCE_DIR}/src/local.cmake")
include("${CMAKE_SOURCE_DIR}/test/local.cmake")

# EXECUTABLE
# ---------------------------------------------------------------------------
add_executable(ic main.cpp)
target_link_libraries(ic ${PROJECT_NAME} SIMDcomp)

# TEST
# ---------------------------------------------------------------------------

add_executable(ic_test test/test.cpp ${TEST_SRC})
target_link_libraries(ic_test ${PROJECT_NAME} SIMDcomp gtest)
enable_testing()
add_test(${PROJECT_NAME} ic_test)