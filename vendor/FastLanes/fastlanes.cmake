######################
# Integer Compression #
#######################

include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
    fastlanes_src
    PREFIX "vendor/FastLanes"
    GIT_REPOSITORY "https://github.com/cwida/FastLanes.git"
    GIT_TAG 6f5651d7691b92f5086e040b50bad220d02c4acd
    TIMEOUT 10
    CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/vendor/FastLanes
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(fastlanes_src BINARY_DIR SOURCE_DIR)

set(FASTLANES_INCLUDE_DIR ${SOURCE_DIR}/src/include)  
set(FASTLANES_LIBRARY_PATH ${BINARY_DIR}/src/libFastLanes.a)

file(MAKE_DIRECTORY ${FASTLANES_INCLUDE_DIR})

add_library(fastlanes STATIC IMPORTED)
add_dependencies(fastlanes fastlanes_src)

set_property(TARGET fastlanes PROPERTY IMPORTED_LOCATION ${FASTLANES_LIBRARY_PATH})
set_property(TARGET fastlanes APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FASTLANES_INCLUDE_DIR})
