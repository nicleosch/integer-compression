#######################
# Integer Compression #
#######################

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    snappy_src
    PREFIX "vendor/snappy"
    GIT_REPOSITORY "https://github.com/google/snappy.git"
    GIT_TAG 6af9287fbdb913f0794d0148c6aa43b58e63c8e3
    TIMEOUT 10
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/vendor/snappy
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    INSTALL_DIR ${CMAKE_BINARY_DIR}/vendor/snappy
)

ExternalProject_Get_Property(snappy_src BINARY_DIR SOURCE_DIR)

set(SNAPPY_INCLUDE_DIR ${CMAKE_BINARY_DIR}/vendor/snappy/include)
set(SNAPPY_LIBRARY_PATH ${CMAKE_BINARY_DIR}/vendor/snappy/lib/libsnappy.a)

file(MAKE_DIRECTORY ${SNAPPY_INCLUDE_DIR})

add_library(snappy STATIC IMPORTED)
add_dependencies(snappy snappy_src)

set_property(TARGET snappy PROPERTY IMPORTED_LOCATION ${SNAPPY_LIBRARY_PATH})
set_property(TARGET snappy APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SNAPPY_INCLUDE_DIR})