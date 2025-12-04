import argparse
import torch
import time
import os

def run_with_cmake():
    lib_path = os.path.join(os.path.dirname(__file__), "build", "libadd2.so")
    torch.ops.load_library(lib_path)
    n = 256
    a = torch.ones((n, n), dtype=torch.float32, device='cuda')
    b = torch.arange(n*n, dtype=torch.float32, device='cuda').reshape(n, n)
    c = torch.empty_like(a)

    torch.cuda.synchronize()
    t0 = time.time()
    torch.ops.add2.torch_launch_add2(c, a, b, n)
    torch.cuda.synchronize()
    t1 = time.time()
    print("Cuda time: ", (t1 - t0) * 1e6, "us")
    assert torch.allclose(c, a + b)
    print("Kernel test passed.")

    torch.cuda.synchronize()
    t2 = time.time()
    c_ref = a + b
    torch.cuda.synchronize()
    t3 = time.time()
    print("Torch time:", (t3 - t2) * 1e6, "us")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=str, default='cmake')
    args = parser.parse_args()
    if args.compiler == 'cmake':
        run_with_cmake()
