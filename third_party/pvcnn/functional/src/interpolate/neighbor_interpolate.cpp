#include "neighbor_interpolate.hpp"
#include "neighbor_interpolate.cuh"

#include "../utils.hpp"

std::vector<at::Tensor>
three_nearest_neighbors_interpolate_forward(at::Tensor points_coords,
                                            at::Tensor centers_coords,
                                            at::Tensor centers_features) {
  CHECK_CUDA(points_coords);
  CHECK_CUDA(centers_coords);
  CHECK_CUDA(centers_features);
  CHECK_CONTIGUOUS(points_coords);
  CHECK_CONTIGUOUS(centers_coords);
  CHECK_CONTIGUOUS(centers_features);
  CHECK_IS_FLOAT(points_coords);
  CHECK_IS_FLOAT(centers_coords);
  CHECK_IS_FLOAT(centers_features);

  int b = centers_features.size(0);
  int c = centers_features.size(1);
  int m = centers_features.size(2);
  int n = points_coords.size(2);

  at::Tensor indices = torch::zeros(
      {b, 3, n}, at::device(points_coords.device()).dtype(at::ScalarType::Int));
  at::Tensor weights = torch::zeros(
      {b, 3, n},
      at::device(points_coords.device()).dtype(at::ScalarType::Float));
  at::Tensor output = torch::zeros(
      {b, c, n},
      at::device(centers_features.device()).dtype(at::ScalarType::Float));

  three_nearest_neighbors_interpolate(
      b, c, m, n, points_coords.data_ptr<float>(),
      centers_coords.data_ptr<float>(), centers_features.data_ptr<float>(),
      indices.data_ptr<int>(), weights.data_ptr<float>(),
      output.data_ptr<float>());
  return {output, indices, weights};
}

at::Tensor three_nearest_neighbors_interpolate_backward(at::Tensor grad_y,
                                                        at::Tensor indices,
                                                        at::Tensor weights,
                                                        const int m) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CUDA(weights);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(weights);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_FLOAT(weights);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  at::Tensor grad_x = torch::zeros(
      {b, c, m}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  three_nearest_neighbors_interpolate_grad(
      b, c, n, m, grad_y.data_ptr<float>(), indices.data_ptr<int>(),
      weights.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}
