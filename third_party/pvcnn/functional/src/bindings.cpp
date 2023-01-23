#include <pybind11/pybind11.h>

#include "ball_query/ball_query.hpp"
#include "grouping/grouping.hpp"
#include "interpolate/neighbor_interpolate.hpp"
#include "interpolate/trilinear_devox.hpp"
#include "sampling/sampling.hpp"
#include "voxelization/vox.hpp"

PYBIND11_MODULE(_pvcnn_backend, m) {
  m.def("gather_features_forward", &gather_features_forward,
        "Gather Centers' Features forward (CUDA)");
  m.def("gather_features_backward", &gather_features_backward,
        "Gather Centers' Features backward (CUDA)");
  m.def("furthest_point_sampling", &furthest_point_sampling_forward,
        "Furthest Point Sampling (CUDA)");
  m.def("ball_query", &ball_query_forward, "Ball Query (CUDA)");
  m.def("grouping_forward", &grouping_forward,
        "Grouping Features forward (CUDA)");
  m.def("grouping_backward", &grouping_backward,
        "Grouping Features backward (CUDA)");
  m.def("three_nearest_neighbors_interpolate_forward",
        &three_nearest_neighbors_interpolate_forward,
        "3 Nearest Neighbors Interpolate forward (CUDA)");
  m.def("three_nearest_neighbors_interpolate_backward",
        &three_nearest_neighbors_interpolate_backward,
        "3 Nearest Neighbors Interpolate backward (CUDA)");

  m.def("trilinear_devoxelize_forward", &trilinear_devoxelize_forward,
        "Trilinear Devoxelization forward (CUDA)");
  m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward,
        "Trilinear Devoxelization backward (CUDA)");
  m.def("avg_voxelize_forward", &avg_voxelize_forward,
        "Voxelization forward with average pooling (CUDA)");
  m.def("avg_voxelize_backward", &avg_voxelize_backward,
        "Voxelization backward (CUDA)");
}
