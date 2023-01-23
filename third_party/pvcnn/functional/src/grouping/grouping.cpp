#include "grouping.hpp"
#include "grouping.cuh"

#include "../utils.hpp"

at::Tensor grouping_forward(at::Tensor features, at::Tensor indices) {
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
  int u = indices.size(2);
  at::Tensor output = torch::zeros(
      {b, c, m, u}, at::device(features.device()).dtype(at::ScalarType::Float));
  grouping(b, c, n, m, u, features.data_ptr<float>(), indices.data_ptr<int>(),
           output.data_ptr<float>());
  return output;
}

at::Tensor grouping_backward(at::Tensor grad_y, at::Tensor indices,
                             const int n) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int m = indices.size(1);
  int u = indices.size(2);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  grouping_grad(b, c, n, m, u, grad_y.data_ptr<float>(),
                indices.data_ptr<int>(), grad_x.data_ptr<float>());
  return grad_x;
}
