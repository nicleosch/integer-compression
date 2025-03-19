#######################
# Integer Compression #
#######################

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
        SIMDcomp_src
        PREFIX "vendor/SIMDcomp"
        GIT_REPOSITORY "https://github.com/fast-pack/simdcomp.git"
        GIT_TAG 009c67807670d16f8984c0534aef0e630e5465a4
        TIMEOUT 10
        CONFIGURE_COMMAND ""
        BUILD_COMMAND make libsimdcomp.a
        BUILD_IN_SOURCE 1
        UPDATE_COMMAND ""
        INSTALL_COMMAND ""
)

ExternalProject_Get_Property(SIMDcomp_src source_dir)

set(SIMDCOMP_INCLUDE_DIR ${source_dir}/include)
set(SIMDCOMP_LIBRARY_PATH ${source_dir}/libsimdcomp.a)

file(MAKE_DIRECTORY ${SIMDCOMP_INCLUDE_DIR})

add_library(SIMDcomp STATIC IMPORTED)
add_dependencies(SIMDcomp SIMDcomp_src)

set_property(TARGET SIMDcomp PROPERTY IMPORTED_LOCATION ${SIMDCOMP_LIBRARY_PATH})
set_property(TARGET SIMDcomp APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${SIMDCOMP_INCLUDE_DIR})
