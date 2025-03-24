#######################
# Integer Compression #
#######################

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    lz4_src
    PREFIX "vendor/lz4"
    GIT_REPOSITORY "https://github.com/lz4/lz4.git"
    GIT_TAG 9d53d8bb6c4120345a0966e5d8b16d7def1f32c5
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(lz4_src source_dir)

set(LZ4_INCLUDE_DIR ${source_dir}/lib)
set(LZ4_LIBRARY_PATH ${source_dir}/lib/liblz4.a)

file(MAKE_DIRECTORY ${LZ4_INCLUDE_DIR})

add_library(lz4 STATIC IMPORTED)
add_dependencies(lz4 lz4_src)

set_property(TARGET lz4 PROPERTY IMPORTED_LOCATION ${LZ4_LIBRARY_PATH})
set_property(TARGET lz4 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${LZ4_INCLUDE_DIR})