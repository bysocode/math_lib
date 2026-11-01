cmake_minimum_required(VERSION 3.16)

project(mathlib
    VERSION 0.1.0
    DESCRIPTION "A small, dependency-free C++ matrix library"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(MATHLIB_BUILD_TESTS "Build mathlib tests" ON)
option(MATHLIB_BUILD_EXAMPLES "Build mathlib examples" ON)

# ---- library -----------------------------------------------------------

add_library(mathlib src/matrix.cpp)
add_library(mathlib::mathlib ALIAS mathlib)

target_include_directories(mathlib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(mathlib PUBLIC cxx_std_17)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mathlib PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
    target_compile_options(mathlib PRIVATE /W4)
endif()

# ---- tests and examples ------------------------------------------------

if(MATHLIB_BUILD_TESTS)
    enable_testing()

    add_executable(mathlib_tests tests/test_matrix.cpp)
    target_link_libraries(mathlib_tests PRIVATE mathlib::mathlib)

    add_test(NAME mathlib_tests COMMAND mathlib_tests)
endif()

if(MATHLIB_BUILD_EXAMPLES)
    add_executable(mathlib_demo examples/demo.cpp)
    target_link_libraries(mathlib_demo PRIVATE mathlib::mathlib)
endif()
