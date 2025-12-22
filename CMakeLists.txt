cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(add2_project LANGUAGES CXX CUDA)

# 尝试自动查找 Torch 路径
execute_process(
    COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_CMAKE_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
    message(STATUS "Found Torch CMake path: ${TORCH_CMAKE_PATH}")
endif()

# 自动查找 PyTorch 的 CMake 配置文件
find_package(Torch REQUIRED)

# 查找 torch_python 库 (解决 undefined symbol 问题)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_CMAKE_PATH}/../../lib")

# 查找 Python3 开发库 (Python.h)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 添加头文件目录
include_directories(include)

# 定义源文件
set(SRCS kernel/add2_ops.cpp kernel/add2_kernel.cu)

# 创建一个名为 add2 的共享库
add_library(add2 SHARED ${SRCS})

# 链接 PyTorch 库和 Python 库
target_link_libraries(add2 PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python3::Python)

# C++17 标准是 PyTorch 扩展的推荐标准
set_property(TARGET add2 PROPERTY CXX_STANDARD 17)