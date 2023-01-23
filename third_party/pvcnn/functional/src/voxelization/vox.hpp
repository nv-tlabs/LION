#ifndef _VOX_HPP
#define _VOX_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> avg_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution);

at::Tensor avg_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt);

#endif
