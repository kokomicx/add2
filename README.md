# Custom CUDA Operator for PyTorch: add2

This project demonstrates how to implement a custom CUDA kernel for matrix addition and integrate it into PyTorch using C++ extensions. It supports three build/run modes: JIT, Setup.py, and CMake.

## Project Structure

- `kernel/add2_kernel.cu`: The CUDA kernel implementation.
- `kernel/add2_ops.cpp`: The C++ wrapper for PyTorch binding.
- `include/add2.h`: Header file declaring the C++ function.
- `run_time.py`: Python script to compile/load the operator and run benchmarks.
- `setup.py`: Script for installing the extension via setuptools.
- `CMakeLists.txt`: Build configuration for CMake.

## Prerequisites

- Python 3.x
- PyTorch (with CUDA support)
- CMake (for CMake mode)
- NVIDIA CUDA Toolkit (nvcc)
- Ninja (optional, speeds up JIT compilation)

## How to Build and Run

You can run the benchmark script `run_time.py` using one of the following methods.

### 1. JIT Compilation (Just-In-Time)

PyTorch compiles the C++/CUDA code on the fly when the script runs. This is the simplest method for development.

```bash
python3 run_time.py --compiler jit
```

### 2. Setup.py Installation

Install the extension as a Python package.

1.  **Install the package:**

    ```bash
    python3 setup.py install
    ```

2.  **Run the script:**

    ```bash
    python3 run_time.py --compiler setup
    ```

### 3. CMake Compilation

Pre-compile the shared library using CMake. This is useful for C++ heavy projects or deployment.

1.  **Build the library:**

    ```bash
    mkdir -p build
    cd build
    cmake ..
    make
    cd ..
    ```

2.  **Run the script:**

    ```bash
    python3 run_time.py --compiler cmake
    ```

## Performance Benchmark

The `run_time.py` script compares the performance of the custom CUDA kernel against PyTorch's native addition operator. It also verifies the correctness of the results.

Expected output includes:
- Correctness check (Kernel test passed/FAILED)
- Average execution time for the custom CUDA kernel
- Average execution time for the native PyTorch operator
