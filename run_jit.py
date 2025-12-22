# run_jit.py
from torch.utils.cpp_extension import load
import torch

# 加载并编译
# PyTorch 会在 ~/.cache/torch_extensions 目录下缓存编译结果
# 只有在源文件发生改变时才会重新编译
cuda_module = load(name="add2",
                   extra_include_paths=["include"], # 头文件路径
                   sources=["kernel/add2_ops.cpp", "kernel/add2_kernel.cu"],
                   verbose=True) # 打印详细编译日志

# 准备数据 (示例)
n = 1024
a = torch.randn(n, n, device='cuda')
b = torch.randn(n, n, device='cuda')
c = torch.empty_like(a)

# 调用
cuda_module.torch_launch_add2(c, a, b)
print("JIT compilation and execution finished.")