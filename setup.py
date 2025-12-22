# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add2",  # 包名
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "add2", # 模块名
            ["kernel/add2_ops.cpp", "kernel/add2_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)