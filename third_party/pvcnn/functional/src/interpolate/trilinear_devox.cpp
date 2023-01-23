#include "trilinear_devox.hpp"
#include "trilinear_devox.cuh"

#include "../utils.hpp"

/*
  Function: trilinear devoxelization (forward)
  Args:
    r        : voxel resolution
    trainig  : whether is training mode
    coords   : the coordinates of points, FloatTensor[b, 3, n]
    features : features, FloatTensor[b, c, s], s = r ** 3
  Return:
    outs : outputs, FloatTensor[b, c, n]
    inds : the voxel coordinates of point cube, IntTensor[b, 8, n]
    wgts : weight for trilinear interpolation, FloatTensor[b, 8, n]
*/
std::vector<at::Tensor>
trilinear_devoxelize_forward(const int r, const bool is_training,
                             const at::Tensor coords,
                             const at::Tensor features) {
  CHECK_CUDA(features);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_FLOAT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = coords.size(2);
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor outs = torch::zeros(
      {b, c, n}, at::device(features.device()).dtype(at::ScalarType::Float));
  if (is_training) {
    at::Tensor inds = torch::zeros(
        {b, 8, n}, at::device(features.device()).dtype(at::ScalarType::Int));
    at::Tensor wgts = torch::zeros(
        {b, 8, n}, at::device(features.device()).dtype(at::ScalarType::Float));
    trilinear_devoxelize(b, c, n, r, r2, r3, true, coords.data_ptr<float>(),
                         features.data_ptr<float>(), inds.data_ptr<int>(),
                         wgts.data_ptr<float>(), outs.data_ptr<float>());
    return {outs, inds, wgts};
  } else {
    at::Tensor inds = torch::zeros(
        {1}, at::device(features.device()).dtype(at::ScalarType::Int));
    at::Tensor wgts = torch::zeros(
        {1}, at::device(features.device()).dtype(at::ScalarType::Float));
    trilinear_devoxelize(b, c, n, r, r2, r3, false, coords.data_ptr<float>(),
                         features.data_ptr<float>(), inds.data_ptr<int>(),
                         wgts.data_ptr<float>(), outs.data_ptr<float>());
    return {outs, inds, wgts};
  }
}

/*
  Function: trilinear devoxelization (backward)
  Args:
    grad_y  : grad outputs, FloatTensor[b, c, n]
    indices : the voxel coordinates of point cube, IntTensor[b, 8, n]
    weights : weight for trilinear interpolation, FloatTensor[b, 8, n]
    r       : voxel resolution
  Return:
    grad_x     : grad inputs, FloatTensor[b, c, s], s = r ** 3
*/
at::Tensor trilinear_devoxelize_backward(const at::Tensor grad_y,
                                         const at::Tensor indices,
                                         const at::Tensor weights,
                                         const int r) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(weights);
  CHECK_CUDA(indices);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(indices);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_FLOAT(weights);
  CHECK_IS_INT(indices);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int n = grad_y.size(2);
  int r3 = r * r * r;
  at::Tensor grad_x = torch::zeros(
      {b, c, r3}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  trilinear_devoxelize_grad(b, c, n, r3, indices.data_ptr<int>(),
                            weights.data_ptr<float>(), grad_y.data_ptr<float>(),
                            grad_x.data_ptr<float>());
  return grad_x;
}
