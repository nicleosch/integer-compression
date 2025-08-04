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
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=address)
    add_link_options(-fsanitize=undefined)
endif()

# PATHS
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/bench)

# DEPENDENCIES
# ---------------------------------------------------------------------------
set(VENDOR_DIR "${CMAKE_SOURCE_DIR}/vendor/")

include("${VENDOR_DIR}/SIMDcomp/simdcomp.cmake")
include("${VENDOR_DIR}/GoogleTest/googletest.cmake")
include("${VENDOR_DIR}/LZ4/lz4.cmake")
include("${VENDOR_DIR}/BtrBlocks/btrblocks.cmake")
include("${VENDOR_DIR}/zstd/zstd.cmake")
include("${VENDOR_DIR}/snappy/snappy.cmake")
include("${VENDOR_DIR}/FastLanes/fastlanes.cmake")

include_directories(
    ${SIMDCOMP_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIR}
    ${BTRBLOCKS_INCLUDE_DIR}
    ${ZSTD_INCLUDE_DIR}
    ${SNAPPY_INCLUDE_DIR}
    ${FASTLANES_INCLUDE_DIR}
)

# COMPONENTS
# ---------------------------------------------------------------------------
include("${CMAKE_SOURCE_DIR}/bench/local.cmake")
include("${CMAKE_SOURCE_DIR}/src/local.cmake")
include("${CMAKE_SOURCE_DIR}/test/local.cmake")

# EXECUTABLE
# ---------------------------------------------------------------------------
add_executable(ic main.cpp)
target_link_libraries(ic ${PROJECT_NAME} SIMDcomp lz4 btrblocks zstd snappy fastlanes)

# BENCHMARKS
# ---------------------------------------------------------------------------
add_executable(ic_bench bench/Benchmarks.cpp ${BENCH_SRC})
target_link_libraries(ic_bench ${PROJECT_NAME} SIMDcomp lz4 btrblocks zstd snappy)

# TEST
# ---------------------------------------------------------------------------

add_executable(ic_test test/test.cpp ${TEST_SRC})
target_link_libraries(ic_test ${PROJECT_NAME} gtest SIMDcomp lz4 btrblocks zstd snappy)
enable_testing()
add_test(${PROJECT_NAME} ic_test)