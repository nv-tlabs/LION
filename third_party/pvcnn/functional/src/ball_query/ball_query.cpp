#include "ball_query.hpp"
#include "ball_query.cuh"

#include "../utils.hpp"

at::Tensor ball_query_forward(at::Tensor centers_coords,
                              at::Tensor points_coords, const float radius,
                              const int num_neighbors) {
  CHECK_CUDA(centers_coords);
  CHECK_CUDA(points_coords);
  CHECK_CONTIGUOUS(centers_coords);
  CHECK_CONTIGUOUS(points_coords);
  CHECK_IS_FLOAT(centers_coords);
  CHECK_IS_FLOAT(points_coords);

  int b = centers_coords.size(0);
  int m = centers_coords.size(2);
  int n = points_coords.size(2);

  at::Tensor neighbors_indices = torch::zeros(
      {b, m, num_neighbors},
      at::device(centers_coords.device()).dtype(at::ScalarType::Int));

  ball_query(b, n, m, radius * radius, num_neighbors,
             centers_coords.data_ptr<float>(),
             points_coords.data_ptr<float>(),
             neighbors_indices.data_ptr<int>());

  return neighbors_indices;
}
