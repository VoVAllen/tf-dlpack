cmake_minimum_required(VERSION 3.5)

project(tfdlpack C CXX)

add_definitions(-std=c++11 -fPIC)
include(cmake/util/FindCUDA.cmake)	

if(NOT PYTHON_EXECUTABLE)
  execute_process(
    COMMAND bash -c "which python"
    OUTPUT_VARIABLE PYTHON_EXECUTABLE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")

# For tf header
execute_process(
  COMMAND bash -c "${PYTHON_EXECUTABLE} -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))'"
  OUTPUT_VARIABLE TF_CFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TF_CFLAGS = ${TF_CFLAGS}")

execute_process(
  COMMAND bash -c "${PYTHON_EXECUTABLE} -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))'"
  OUTPUT_VARIABLE TF_LFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TF_LFLAGS = ${TF_LFLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 ${TF_CFLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_CFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_CFLAGS}")

find_cuda(ON REQUIRED)	
if(NOT CUDA_FOUND)	
  message(FATAL_ERROR "Cannot find CUDA.")	
endif()

include_directories(third_party/dlpack/include)

file(GLOB SRC
  src/dlpack_op.cc
  src/to_dlpack_kernel.cc
  src/get_device_and_dtype_kernel.cc
  src/from_dlpack_kernel.cc
)

cuda_add_library(tfdlpack SHARED ${SRC})
target_link_libraries(tfdlpack ${TF_LFLAGS})

install(TARGETS tfdlpack DESTINATION tfdlpack)
