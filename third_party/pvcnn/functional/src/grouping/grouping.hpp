#ifndef _GROUPING_HPP
#define _GROUPING_HPP

#include <torch/extension.h>

at::Tensor grouping_forward(at::Tensor features, at::Tensor indices);
at::Tensor grouping_backward(at::Tensor grad_y, at::Tensor indices,
                             const int n);

#endif
