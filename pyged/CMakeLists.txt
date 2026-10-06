cmake_minimum_required(VERSION 3.16.1)

project(pyged)

find_package(pybind11)
pybind11_add_module(pyged src/pyged.cpp)

if(NOT DEFINED ENV{GUROBI_HOME})
	message(FATAL_ERROR "GUROBI_HOME is not set. Load the Jean Zay gurobi module first.")
endif()
set(GUROBI_HOME "$ENV{GUROBI_HOME}" CACHE PATH "Gurobi installation root")
find_path(GUROBI_INCLUDE_DIR gurobi_c.h HINTS "${GUROBI_HOME}/include" REQUIRED)
find_library(GUROBI_CXX_LIBRARY NAMES gurobi_c++ HINTS "${GUROBI_HOME}/lib" REQUIRED NO_DEFAULT_PATH)
file(GLOB GUROBI_C_LIBRARIES "${GUROBI_HOME}/lib/libgurobi[0-9][0-9]*${CMAKE_SHARED_LIBRARY_SUFFIX}")
list(SORT GUROBI_C_LIBRARIES COMPARE NATURAL ORDER ASCENDING)
list(LENGTH GUROBI_C_LIBRARIES GUROBI_LIBRARY_COUNT)
if(GUROBI_LIBRARY_COUNT EQUAL 0)
	message(FATAL_ERROR "No versioned Gurobi C library found below ${GUROBI_HOME}/lib")
endif()
math(EXPR GUROBI_LAST_LIBRARY "${GUROBI_LIBRARY_COUNT} - 1")
list(GET GUROBI_C_LIBRARIES ${GUROBI_LAST_LIBRARY} GUROBI_C_LIBRARY)
message(STATUS "Using Gurobi include: ${GUROBI_INCLUDE_DIR}")
message(STATUS "Using Gurobi C++ library: ${GUROBI_CXX_LIBRARY}")
message(STATUS "Using Gurobi C library: ${GUROBI_C_LIBRARY}")

add_custom_command(TARGET pyged POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib
	COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pyged> ${CMAKE_SOURCE_DIR}/lib/
)

target_compile_options(pyged PUBLIC
	-fpermissive
)

target_include_directories(pyged PUBLIC
	${CMAKE_SOURCE_DIR}/ext/gedlib
	${CMAKE_SOURCE_DIR}/ext/gedlib/ext/boost_1_82_0
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
	doublefann
	svm
	nomad
	${GUROBI_CXX_LIBRARY}
	${GUROBI_C_LIBRARY}
)

set_target_properties(pyged PROPERTIES BUILD_RPATH "${GUROBI_HOME}/lib")
