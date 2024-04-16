## ---------------------------------------------------------------------
##
## Copyright (C) 2018 Xiaoyu Wei
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
## THE SOFTWARE.
##
## ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15.0)
project(meshgen11_dealii LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# deal.II
find_package(deal.II REQUIRED
    HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{CONDA_PREFIX})
deal_ii_initialize_cached_variables()

# Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(meshgen11 meshgen.cpp)
set_target_properties(meshgen11 PROPERTIES
    OUTPUT_NAME "meshgen"
    LINK_FLAGS ${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_RELEASE}
    COMPILE_DEFINITIONS ${DEAL_II_USER_DEFINITIONS}
    INTERPROCEDURAL_OPTIMIZATION ON
    VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(meshgen11 PRIVATE ${DEAL_II_INCLUDE_DIRS})
target_link_libraries(meshgen11 PRIVATE ${DEAL_II_TARGET_RELEASE})

add_custom_command(TARGET meshgen11 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "meshgen.*.so"
    "meshgen.so"
    COMMENT "Copied meshgen lib to meshgen.so")
