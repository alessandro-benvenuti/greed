cmake_minimum_required(VERSION 3.16.1)

project(pyged LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(pyged src/pyged.cpp)

if(NOT DEFINED ENV{GUROBI_HOME})
	message(FATAL_ERROR "GUROBI_HOME is not set. Load the Jean Zay gurobi module first.")
endif()
set(GUROBI_HOME "$ENV{GUROBI_HOME}" CACHE PATH "Gurobi installation root")
if(NOT DEFINED ENV{BOOST_ROOT})
	message(FATAL_ERROR "BOOST_ROOT is not set. Load the Jean Zay boost/1.86.0 module first.")
endif()
find_path(BOOST_INCLUDE_DIR boost/version.hpp HINTS "$ENV{BOOST_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_path(GUROBI_INCLUDE_DIR gurobi_c.h HINTS "${GUROBI_HOME}/include" REQUIRED NO_DEFAULT_PATH)
find_library(GUROBI_CXX_LIBRARY NAMES gurobi_c++ HINTS "${GUROBI_HOME}/lib" REQUIRED NO_DEFAULT_PATH)
file(GLOB GUROBI_C_LIBRARIES "${GUROBI_HOME}/lib/libgurobi[0-9][0-9]*${CMAKE_SHARED_LIBRARY_SUFFIX}")
list(FILTER GUROBI_C_LIBRARIES EXCLUDE REGEX "_light${CMAKE_SHARED_LIBRARY_SUFFIX}$")
list(SORT GUROBI_C_LIBRARIES COMPARE NATURAL ORDER ASCENDING)
list(LENGTH GUROBI_C_LIBRARIES GUROBI_LIBRARY_COUNT)
if(GUROBI_LIBRARY_COUNT EQUAL 0)
	message(FATAL_ERROR "No versioned Gurobi C library found below ${GUROBI_HOME}/lib")
endif()
math(EXPR GUROBI_LAST_LIBRARY "${GUROBI_LIBRARY_COUNT} - 1")
list(GET GUROBI_C_LIBRARIES ${GUROBI_LAST_LIBRARY} GUROBI_C_LIBRARY)
message(STATUS "Using Gurobi include: ${GUROBI_INCLUDE_DIR}")
message(STATUS "Using Boost include: ${BOOST_INCLUDE_DIR}")
message(STATUS "Using Gurobi C++ library: ${GUROBI_CXX_LIBRARY}")
message(STATUS "Using Gurobi C library: ${GUROBI_C_LIBRARY}")

find_library(FANN_LIBRARY NAMES doublefann
	HINTS "${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/lib" REQUIRED NO_DEFAULT_PATH)
find_library(SVM_LIBRARY NAMES svm
	HINTS "${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22" REQUIRED NO_DEFAULT_PATH)
find_library(NOMAD_LIBRARY NAMES nomad
	HINTS "${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/lib" REQUIRED NO_DEFAULT_PATH)

add_custom_command(TARGET pyged POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pyged> ${CMAKE_SOURCE_DIR}/lib/
)

target_compile_options(pyged PUBLIC
	-fpermissive
)

target_include_directories(pyged PUBLIC
	${CMAKE_SOURCE_DIR}/ext/gedlib
	${BOOST_INCLUDE_DIR}
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/eigen.3.3.4/Eigen
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/src
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/ext/sgtelib/src
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/lsape.5/include
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/include
	${GUROBI_INCLUDE_DIR}
)

target_link_directories(pyged PUBLIC
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/lib
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/lib
	${GUROBI_HOME}/lib
)

target_link_libraries(pyged PUBLIC
	${FANN_LIBRARY}
	${SVM_LIBRARY}
	${NOMAD_LIBRARY}
	${GUROBI_CXX_LIBRARY}
	${GUROBI_C_LIBRARY}
)

set(GEDLIB_RUNTIME_LIBRARY_DIRS
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/nomad.3.8.1/lib
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/libsvm.3.22
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/fann.2.2.0/lib
	${GUROBI_HOME}/lib
)
set_target_properties(pyged PROPERTIES
	BUILD_RPATH "${GEDLIB_RUNTIME_LIBRARY_DIRS}"
	INSTALL_RPATH "${GEDLIB_RUNTIME_LIBRARY_DIRS}"
)
