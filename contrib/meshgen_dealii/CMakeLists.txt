## ---------------------------------------------------------------------
##
## Copyright (C) 2017 - 2018 Xiaoyu Wei
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
PROJECT(meshgen_dealii)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

#set(PYTHON_DIR "/home/xywei/miniconda3/envs/inteq/")
#find_package(PythonLibs 3 REQUIRED
#  HINTS ${PYTHON_DIR})
set(PYTHON_INCLUDE_DIRS "/usr/include/python3.6m/" CACHE STRING "Python include dir")
set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.6m.so" CACHE STRING "Python shared lib fie")
include_directories(${PYTHON_INCLUDE_DIRS})

option(UseCustomBuiltBoost "Use custom-built boost" ON)

if(UseCustomBuiltBoost)
  set(BOOST_ROOT "/home/xywei/apps" CACHE STRING "Boost dir")
  find_package(Boost 1.65.1 COMPONENTS python3 REQUIRED)
else()
  # workaround for Debian
  set(BOOST_ROOT "/usr/" CACHE STRING "Boost dir")
  set(BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu/" CACHE STRING "Boost lib dir")

  set(Boost_DIR "/usr/" CACHE PATH "Boost dir")
  set(Boost_INCLUDE_DIR "/usr/include/boost" CACHE PATH "Boost include dir")
  set(Boost_PYTHON3_LIBRARY_DEBUG "/usr/lib/x86_64-linux-gnu/libboost_python-py36.so" CACHE FILEPATH "Boost python lib")
  set(Boost_PYTHON3_LIBRARY_RELEASE "/usr/lib/x86_64-linux-gnu/libboost_python-py36.so" CACHE FILEPATH "Boost python lib")
  include_directories(${Boost_INCLUDE_DIRS})
endif()

# find_package(Boost 1.62.0 COMPONENTS python3 REQUIRED)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
endif()
set(NUMPY_LIBRARIES "/home/xywei/apps/lib/libboost_numpy3.so" CACHE STRING "Boost numpy shared lib file")

set(DEAL_II_DIR "/home/xywei/apps" CACHE STRING "Deal.II dir")
#FIND_PACKAGE(deal.II 8.5.0 REQUIRED
FIND_PACKAGE(deal.II 8.4.2 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR} $ENV{CONDA_PREFIX}
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

ADD_LIBRARY(meshgen SHARED meshgen.cpp)
set_target_properties(meshgen PROPERTIES PREFIX "")
set_target_properties(meshgen PROPERTIES OUTPUT_NAME "meshgen")
TARGET_LINK_LIBRARIES(meshgen LINK_PUBLIC ${Boost_LIBRARIES})
TARGET_LINK_LIBRARIES(meshgen LINK_PUBLIC ${PYTHON_LIBRARIES})
TARGET_LINK_LIBRARIES(meshgen LINK_PUBLIC ${NUMPY_LIBRARIES})
DEAL_II_SETUP_TARGET(meshgen)
