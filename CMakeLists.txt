cmake_minimum_required (VERSION 2.6)

# See https://stackoverflow.com/a/31010221/23845
if(CMAKE_VERSION VERSION_LESS "3.1")
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  if(NOT COMPILER_SUPPORTS_CXX11)
    MESSAGE(FATAL_ERROR "Your compiler does not support c++11")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

project (TensorComprehensions C CXX)

# Display compiler flags for build modes: Release/Debug
message(STATUS "BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION is ${CMAKE_CXX_COMPILER_VERSION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.4")
  # we officially support these compiler flags for compiler version 5.4 or less
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Werror -Wno-sign-compare")
endif()
message(STATUS "CMAKE_CXX_FLAGS is ${CMAKE_CXX_FLAGS}")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
endif()
message(STATUS "CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")

#add_subdirectory(third-party/pybind11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTC_DIR=\"\\\"${TC_DIR}\\\"\" ")


# RPATH - how to properly handle rpath https://cmake.org/Wiki/CMake_RPATH_handling
# RPATH - a list of directories which is linked into the executable, supported on most UNIX systems.
# By default if you don't change any RPATH related settings, CMake will link the
# executables and shared libraries with full RPATH to all used libraries in
# the build tree. When installing, it will clear the RPATH of these targets so
# they are installed with an empty RPATH. The following settings are recommended

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# don't add the automatically determined parts of the RPATH which point to
# directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
  # the RPATH to be used when installing
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_PREFIX}/lib64")
endif()

set(GIT_EXECUTABLE git)

# If we don't set this, manually specified include_directories in our
# cmake files will be placed after system headers, which means that
# we will pick headers from a conda environment before our own. That's
# very confusing!
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)

# Make it possible to start cmake directly in an appropriately conda environment
if(NOT DEFINED CMAKE_PREFIX_PATH AND DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

# NB: If the finding logic here gets complicated, make a proper FindXXX.cmake
# module and use find_package

# Normal dependencies


# isl
find_path(ISL_INCLUDE_DIR NAMES isl)
include_directories(${ISL_INCLUDE_DIR})
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
find_library(ISL_LIBRARIES isl PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found isl: ${ISL_LIBRARIES}")

# dlpack
find_path(ISL_INCLUDE_DIR NAMES dlpack)
include_directories(${DLPACK_INCLUDE_DIR})

# protobuf
# TODO: check if this respects CMAKE_PREFIX_PATH
find_package(Protobuf REQUIRED)

################################################################################

# Halide
message(STATUS "Finding Halide")
find_path(HALIDE_INCLUDE_DIR NAMES Halide.h)
message(STATUS "HALIDE_INCLUDE_DIRS: ${HALIDE_INCLUDE_DIR}")
include_directories(${HALIDE_INCLUDE_DIR})
find_library(HALIDE_LIBRARIES NAMES Halide PATHS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found Halide.so file: ${HALIDE_LIBRARIES}")

################################################################################


################################################################################

# git revision
message("include(cmake/GetGitRevisionDescription.cmake)")
include(cmake/GetGitRevisionDescription.cmake)

################################################################################

# gflags
message("Looking for gflags...")
find_library(GFLAGS_LIBRARIES gflags HINTS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found gflags: ${GFLAGS_LIBRARIES}")

# glog
# For some reason, the world wants to redefine the glog macros with the same
# name. For now, we use glog everywhere, also force DMLC/CAFFE2  to use glog
add_definitions(-DDMLC_USE_GLOG)
find_library(GLOG_LIBRARIES glog HINTS ${CMAKE_INSTALL_PREFIX} PATH_SUFFIXES lib lib64)
message(STATUS "Found glog: ${GLOG_LIBRARIES}")

# Uncomment for debugging all the CMake variables
# get_cmake_property(_variableNames VARIABLES)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

include_directories(BEFORE ${PROJECT_SOURCE_DIR})
add_subdirectory(tc)
