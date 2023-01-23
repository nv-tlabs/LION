#include "sampling.hpp"
#include "sampling.cuh"

#include "../utils.hpp"

at::Tensor gather_features_forward(at::Tensor features, at::Tensor indices) {
  CHECK_CUDA(features);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(indices);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int m = indices.size(1);
  at::Tensor output = torch::zeros(
      {b, c, m}, at::device(features.device()).dtype(at::ScalarType::Float));
  gather_features(b, c, n, m, features.data_ptr<float>(),
                  indices.data_ptr<int>(), output.data_ptr<float>());
  return output;
}

at::Tensor gather_features_backward(at::Tensor grad_y, at::Tensor indices,
                                    const int n) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  gather_features_grad(b, c, n, indices.size(1), grad_y.data_ptr<float>(),
                       indices.data_ptr<int>(), grad_x.data_ptr<float>());
  return grad_x;
}

at::Tensor furthest_point_sampling_forward(at::Tensor coords,
                                           const int num_samples) {
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(coords);

  int b = coords.size(0);
  int n = coords.size(2);
  at::Tensor indices = torch::zeros(
      {b, num_samples}, at::device(coords.device()).dtype(at::ScalarType::Int));
  at::Tensor distances = torch::full(
      {b, n}, 1e38f, at::device(coords.device()).dtype(at::ScalarType::Float));
  furthest_point_sampling(b, n, num_samples, coords.data_ptr<float>(),
                          distances.data_ptr<float>(), indices.data_ptr<int>());
  return indices;
}
