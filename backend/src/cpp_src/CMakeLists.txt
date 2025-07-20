cmake_minimum_required(VERSION 3.10)
project(LSMC_AAD_Pricer CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add source files for the core AAD types and operations
add_library(aad_core STATIC
    aad_types.h
    aad_ops.h aad_ops.cpp
    matrix_ops_aad.h matrix_ops_aad.cpp
)

# Add source files for utilities
add_library(utils STATIC
    utils_cpp.h utils_cpp.cpp
)

# Add source files for the LSMC AAD pricer itself
add_library(lsmc_pricer STATIC
    lsmc_aad_pricer.h lsmc_aad_pricer.cpp
)

# Add source files for the FFI interface
# This will be compiled into a shared library that Node.js can load
add_library(lsmc_aad_interface SHARED
    lsmc_aad_interface.h
    lsmc_aad_interface.cpp
)

# Link libraries
target_link_libraries(lsmc_aad_interface
    PUBLIC
        aad_core
        utils
        lsmc_pricer
)

# Set output directory for the shared library (for easier FFI access)
set_target_properties(lsmc_aad_interface PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Optional: If you plan to add Google Test for unit testing
# enable_testing()
# find_package(GTest CONFIG REQUIRED)
# add_executable(test_lsmc_aad test/test_lsmc_aad.cpp)
# target_link_libraries(test_lsmc_aad GTest::gtest_main)
# add_test(NAME LSMC_AAD_Tests COMMAND test_lsmc_aad)