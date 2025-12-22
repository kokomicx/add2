import argparse
import torch
import time
import os
from torch.utils.cpp_extension import load

# ------------------------------------------------
# 1. 通用的测试与计时函数
# ------------------------------------------------
def run_benchmark(custom_add_func, name="Custom Op"):
    """
    Args:
        custom_add_func: 封装好的可调用算子函数
        name: 打印日志时的名称
    """
    print(f"Testing method: {name}")
    
    n = 1024  # 为了测试更明显，可以适当调大 n，比如 1024 或 2048
    device = 'cuda'
    
    # 初始化数据
    a = torch.rand((n, n), device=device, dtype=torch.float32)
    b = torch.rand((n, n), device=device, dtype=torch.float32)
    c = torch.empty_like(a)

    # --- Warm up (预热) ---
    # GPU 首次调用会有初始化开销，预热可以保证后续计时准确
    for _ in range(10):
        custom_add_func(c, a, b, n)
    torch.cuda.synchronize()

    # --- 自定义算子计时 ---
    t0 = time.time()
    # 循环多次取平均值会更准确
    loop_times = 100
    for _ in range(loop_times):
        custom_add_func(c, a, b, n)
    
    torch.cuda.synchronize()
    t1 = time.time()
    
    avg_cuda_time = (t1 - t0) / loop_times * 1e6
    print(f"Cuda time (avg of {loop_times} runs): {avg_cuda_time:.3f} us")

    # --- 正确性验证 ---
    # 使用 PyTorch 原生加法作为基准
    c_ref = a + b
    # allclose 用于浮点数比较，允许微小误差
    if torch.allclose(c, c_ref, atol=1e-5):
        print("Kernel test passed.")
    else:
        print("Kernel test FAILED!")
        # 打印最大误差以供调试
        print(f"Max error: {(c - c_ref).abs().max().item()}")

    # --- PyTorch 原生算子计时 ---
    # 同样预热
    for _ in range(10):
        _ = a + b
    torch.cuda.synchronize()

    t2 = time.time()
    for _ in range(loop_times):
        c_ref = a + b
    torch.cuda.synchronize()
    t3 = time.time()
    
    avg_torch_time = (t3 - t2) / loop_times * 1e6
    print(f"Torch time (avg of {loop_times} runs): {avg_torch_time:.3f} us")
    print("-" * 30)


# ------------------------------------------------
# 2. 三种加载方式的实现
# ------------------------------------------------

def run_with_jit():
    """
    JIT 模式：运行时动态编译
    """
    # 获取当前文件所在目录，确保路径正确
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 假设源码在 kernel 文件夹下，头文件在 include 文件夹下
    source_files = [
        os.path.join(base_dir, "kernel/add2_ops.cpp"),
        os.path.join(base_dir, "kernel/add2_kernel.cu"),
    ]
    include_dirs = [os.path.join(base_dir, "include")]

    print("Compiling via JIT...")
    # load 函数会自动处理编译和加载
    add2_module = load(
        name="add2_jit",
        sources=source_files,
        extra_include_paths=include_dirs,
        verbose=True
    )
    
    # 返回模块中的函数
    return add2_module.torch_launch_add2

def run_with_setup():
    """
    Setup 模式：假设已经通过 pip install . 或 python setup.py install 安装
    """
    try:
        import add2 # 直接导入包名
    except ImportError:
        raise ImportError("Module 'add2' not found. Please run 'python setup.py install' first.")
    
    return add2.torch_launch_add2

def run_with_cmake():
    """
    CMake 模式：加载编译好的 .so 文件
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 假设 .so 文件在 build 目录下
    lib_path = os.path.join(base_dir, "build", "libadd2.so")
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found at {lib_path}. Please run cmake build first.")

    torch.ops.load_library(lib_path)
    
    # CMake/TORCH_LIBRARY 注册的方式是通过 torch.ops.<namespace>.<func_name> 调用
    # namespace 'add2' 是在 TORCH_LIBRARY(add2, m) 中定义的
    return torch.ops.add2.torch_launch_add2


# ------------------------------------------------
# 3. 主程序入口
# ------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--compiler', 
        type=str, 
        default='jit', 
        choices=['jit', 'setup', 'cmake'],
        help='Choose compilation method: jit, setup, or cmake'
    )
    args = parser.parse_args()

    try:
        if args.compiler == 'jit':
            func = run_with_jit()
            run_benchmark(func, name="JIT Compilation")
            
        elif args.compiler == 'setup':
            func = run_with_setup()
            run_benchmark(func, name="Setup.py Installed")
            
        elif args.compiler == 'cmake':
            func = run_with_cmake()
            run_benchmark(func, name="CMake Build")
            
    except Exception as e:
        print(f"Error running with {args.compiler}: {e}")