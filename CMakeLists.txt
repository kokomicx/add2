cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(add2 LANGUAGES CXX CUDA)

find_package(Python REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH
)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

set(SRCS kernel/add2_ops.cpp kernel/add2_kernel.cu)
add_library(add2 SHARED ${SRCS})

target_link_libraries(add2 "${TORCH_LIBRARIES}")
set_target_properties(add2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

