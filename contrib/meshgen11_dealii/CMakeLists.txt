CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
PROJECT(meshgen11_dealii)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "DEBUG")
  #SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

# set(DEAL_II_DIR "/home/xywei/apps" CACHE STRING "Deal.II dir")
FIND_PACKAGE(deal.II 8.4.2 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Avoid using 3.7
find_package( PythonInterp 3.6 EXACT )
find_package( PythonLibs 3.6 EXACT )

add_subdirectory(pybind11)
pybind11_add_module(meshgen11 meshgen.cpp)
set_target_properties(meshgen11 PROPERTIES PREFIX "")
set_target_properties(meshgen11 PROPERTIES OUTPUT_NAME "meshgen")

if(FALSE)
  # This script does not work on current Debian testing
  DEAL_II_SETUP_TARGET(meshgen11)
else()
  SET(_build "RELEASE")
  STRING(TOLOWER "${CMAKE_BUILD_TYPE}" _cmake_build_type)
  IF("${_cmake_build_type}" MATCHES "debug")
    SET(_build "DEBUG")
  ENDIF()
  SET_PROPERTY(TARGET meshgen11 APPEND PROPERTY
    INCLUDE_DIRECTORIES "${DEAL_II_INCLUDE_DIRS}"
    )
  SET_PROPERTY(TARGET meshgen11 APPEND_STRING PROPERTY
    COMPILE_FLAGS "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_${_build}}"
    )
  SET_PROPERTY(TARGET meshgen11 APPEND_STRING PROPERTY
    LINK_FLAGS " ${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_${_build}}"
    )
  SET_PROPERTY(TARGET meshgen11 APPEND PROPERTY
    COMPILE_DEFINITIONS "${DEAL_II_USER_DEFINITIONS};${DEAL_II_USER_DEFINITIONS_${_build}}"
    )
  GET_PROPERTY(_type TARGET meshgen11 PROPERTY TYPE)
  IF(NOT "${_type}" STREQUAL "OBJECT_LIBRARY")
    TARGET_LINK_LIBRARIES(meshgen11 PRIVATE ${DEAL_II_TARGET_${_build}})
  ENDIF()
endif()

ADD_CUSTOM_COMMAND(TARGET meshgen11 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  "meshgen.*.so"
  "meshgen.so"
  COMMENT "Copied meshgen lib to meshgen.so")
