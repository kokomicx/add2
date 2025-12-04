#include <ATen/ATen.h>
#include <torch/library.h>
#include "../include/add2.h"

void torch_launch_add2(at::Tensor &c,
                       const at::Tensor &a,
                       const at::Tensor &b,
                       int64_t n) {
    launch_add2(c.data_ptr<float>(),
                a.data_ptr<float>(),
                b.data_ptr<float>(),
                static_cast<int>(n));
}

TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}
