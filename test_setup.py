import torch
import add2 # 直接导入编译好的模块

# ... 准备数据 ...
n = 1024
a = torch.randn(n, n, device='cuda')
b = torch.randn(n, n, device='cuda')
c = torch.empty_like(a)


add2.torch_launch_add2(c, a, b)

# ... 检查结果 ...
assert torch.allclose(c, a + b)