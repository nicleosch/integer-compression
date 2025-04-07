#######################
# Integer Compression #
#######################

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    btrblocks_src
    PREFIX "vendor/BtrBlocks"
    GIT_REPOSITORY "https://github.com/pascalginter/btrblocks.git"
    GIT_TAG 0a577c47822d00c635921e6c196f118e16cfa006
    TIMEOUT 10
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/vendor/BtrBlocks
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    BUILD_COMMAND $(MAKE) btrblocks
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(btrblocks_src BINARY_DIR SOURCE_DIR)

set(BTRBLOCKS_INCLUDE_DIR ${SOURCE_DIR}/btrblocks)  
set(BTRBLOCKS_LIBRARY_PATH ${BINARY_DIR}/libbtrblocks.a)

file(MAKE_DIRECTORY ${BTRBLOCKS_INCLUDE_DIR})

add_library(btrblocks STATIC IMPORTED)
add_dependencies(btrblocks btrblocks_src)

set_property(TARGET btrblocks PROPERTY IMPORTED_LOCATION ${BTRBLOCKS_LIBRARY_PATH})
set_property(TARGET btrblocks APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${BTRBLOCKS_INCLUDE_DIR})

# DEPENDENCIES
# ---------------------------------------------------------------------------´

# FSST
set(FSST_INCLUDE_DIR ${BINARY_DIR}/vendor/cwida/fsst/src/fsst_src)
file(MAKE_DIRECTORY ${FSST_INCLUDE_DIR})

add_library(fsst STATIC IMPORTED)
set_property(TARGET fsst PROPERTY IMPORTED_LOCATION ${BINARY_DIR}/vendor/cwida/fsst/src/fsst_src-build/libfsst.a)
set_property(TARGET fsst PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FSST_INCLUDE_DIR})

# Croaring
set(CROARING_INCLUDE_DIR ${BINARY_DIR}/vendor/croaring/include)
file(MAKE_DIRECTORY ${CROARING_INCLUDE_DIR})

add_library(croaring SHARED IMPORTED)
set_property(TARGET croaring PROPERTY IMPORTED_LOCATION ${BINARY_DIR}/vendor/croaring/lib/libroaring.so)
set_property(TARGET croaring PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CROARING_INCLUDE_DIR})

# FastPFOR
set(FASTPFOR_INCLUDE_DIR ${BINARY_DIR}/vendor/lemire/fastpfor/src/fastpfor_src/headers)
file(MAKE_DIRECTORY ${FASTPFOR_INCLUDE_DIR})

add_library(fastpfor STATIC IMPORTED)
set_property(TARGET fastpfor PROPERTY IMPORTED_LOCATION ${BINARY_DIR}/vendor/lemire/fastpfor/src/fastpfor_src-build/libFastPFOR.a)
set_property(TARGET fastpfor PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FASTPFOR_INCLUDE_DIR})

set_property(TARGET btrblocks APPEND PROPERTY INTERFACE_LINK_LIBRARIES 
    fsst 
    croaring 
    fastpfor
)
