#######################
# Integer Compression #
#######################

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    zstd_src
    PREFIX "vendor/zstd"
    GIT_REPOSITORY "https://github.com/facebook/zstd.git"
    GIT_TAG v1.5.6
    TIMEOUT 10
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
    BUILD_IN_SOURCE 1
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(zstd_src source_dir)

set(ZSTD_INCLUDE_DIR ${source_dir}/lib)
set(ZSTD_LIBRARY_PATH ${source_dir}/lib/libzstd.a)

file(MAKE_DIRECTORY ${ZSTD_INCLUDE_DIR})

add_library(zstd STATIC IMPORTED)
add_dependencies(zstd zstd_src)

set_property(TARGET zstd PROPERTY IMPORTED_LOCATION ${ZSTD_LIBRARY_PATH})
set_property(TARGET zstd APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ZSTD_INCLUDE_DIR})