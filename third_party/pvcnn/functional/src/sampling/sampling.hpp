#ifndef _SAMPLING_HPP
#define _SAMPLING_HPP

#include <torch/extension.h>

at::Tensor gather_features_forward(at::Tensor features, at::Tensor indices);
at::Tensor gather_features_backward(at::Tensor grad_y, at::Tensor indices,
                                    const int n);
at::Tensor furthest_point_sampling_forward(at::Tensor coords,
                                           const int num_samples);

#endif
