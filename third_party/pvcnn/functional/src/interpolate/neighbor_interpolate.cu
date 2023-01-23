#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: three nearest neighbors
  Args:
    b   : batch size
    n   : number of points in point clouds
    m   : number of query centers
    points_coords : coordinates of points, FloatTensor[b, 3, n]
    centers_coords: coordinates of centers, FloatTensor[b, 3, m]
    weights       : weights of nearest 3 centers to the point,
                    FloatTensor[b, 3, n]
    indices       : indices of nearest 3 centers to the point,
                    IntTensor[b, 3, n]
*/
__global__ void three_nearest_neighbors_kernel(
    int b, int n, int m, const float *__restrict__ points_coords,
    const float *__restrict__ centers_coords, float *__restrict__ weights,
    int *__restrict__ indices) {
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;
  points_coords += batch_index * 3 * n;
  weights += batch_index * 3 * n;
  indices += batch_index * 3 * n;
  centers_coords += batch_index * 3 * m;

  for (int j = index; j < n; j += stride) {
    float ux = points_coords[j];
    float uy = points_coords[j + n];
    float uz = points_coords[j + n + n];

    double best0 = 1e40, best1 = 1e40, best2 = 1e40;
    int besti0 = 0, besti1 = 0, besti2 = 0;
    for (int k = 0; k < m; ++k) {
      float x = centers_coords[k];
      float y = centers_coords[k + m];
      float z = centers_coords[k + m + m];
      float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
      if (d < best2) {
        best2 = d;
        besti2 = k;
        if (d < best1) {
          best2 = best1;
          besti2 = besti1;
          best1 = d;
          besti1 = k;
          if (d < best0) {
            best1 = best0;
            besti1 = besti0;
            best0 = d;
            besti0 = k;
          }
        }
      }
    }
    best0 = max(min(1e10f, best0), 1e-10f);
    best1 = max(min(1e10f, best1), 1e-10f);
    best2 = max(min(1e10f, best2), 1e-10f);
    float d0d1 = best0 * best1;
    float d0d2 = best0 * best2;
    float d1d2 = best1 * best2;
    float d0d1d2 = 1.0f / (d0d1 + d0d2 + d1d2);
    weights[j] = d1d2 * d0d1d2;
    indices[j] = besti0;
    weights[j + n] = d0d2 * d0d1d2;
    indices[j + n] = besti1;
    weights[j + n + n] = d0d1 * d0d1d2;
    indices[j + n + n] = besti2;
  }
}

/*
  Function: interpolate three nearest neighbors (forward)
  Args:
    b   : batch size
    c   : #channels of features
    m   : number of query centers
    n   : number of points in point clouds
    centers_features: features of centers, FloatTensor[b, c, m]
    indices         : indices of nearest 3 centers to the point,
                      IntTensor[b, 3, n]
    weights         : weights for interpolation, FloatTensor[b, 3, n]
    out             : features of points, FloatTensor[b, c, n]
*/
__global__ void three_nearest_neighbors_interpolate_kernel(
    int b, int c, int m, int n, const float *__restrict__ centers_features,
    const int *__restrict__ indices, const float *__restrict__ weights,
    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  centers_features += batch_index * m * c;
  indices += batch_index * n * 3;
  weights += batch_index * n * 3;
  out += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    float w1 = weights[j];
    float w2 = weights[j + n];
    float w3 = weights[j + n + n];
    int i1 = indices[j];
    int i2 = indices[j + n];
    int i3 = indices[j + n + n];

    out[i] = centers_features[l * m + i1] * w1 +
             centers_features[l * m + i2] * w2 +
             centers_features[l * m + i3] * w3;
  }
}

void three_nearest_neighbors_interpolate(int b, int c, int m, int n,
                                         const float *points_coords,
                                         const float *centers_coords,
                                         const float *centers_features,
                                         int *indices, float *weights,
                                         float *out) {
  three_nearest_neighbors_kernel<<<b, optimal_num_threads(n), 0,
                                   at::cuda::getCurrentCUDAStream()>>>(
      b, n, m, points_coords, centers_coords, weights, indices);
  three_nearest_neighbors_interpolate_kernel<<<
      b, optimal_block_config(n, c), 0, at::cuda::getCurrentCUDAStream()>>>(
      b, c, m, n, centers_features, indices, weights, out);
  CUDA_CHECK_ERRORS();
}

/*
  Function: interpolate three nearest neighbors (backward)
  Args:
    b   : batch size
    c   : #channels of features
    m   : number of query centers
    n   : number of points in point clouds
    grad_y  : grad of features of points, FloatTensor[b, c, n]
    indices : indices of nearest 3 centers to the point, IntTensor[b, 3, n]
    weights : weights for interpolation, FloatTensor[b, 3, n]
    grad_x  : grad of features of centers, FloatTensor[b, c, m]
*/
__global__ void three_nearest_neighbors_interpolate_grad_kernel(
    int b, int c, int n, int m, const float *__restrict__ grad_y,
    const int *__restrict__ indices, const float *__restrict__ weights,
    float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  grad_y += batch_index * n * c;
  indices += batch_index * n * 3;
  weights += batch_index * n * 3;
  grad_x += batch_index * m * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * n; i += stride) {
    const int l = i / n;
    const int j = i % n;
    float w1 = weights[j];
    float w2 = weights[j + n];
    float w3 = weights[j + n + n];
    int i1 = indices[j];
    int i2 = indices[j + n];
    int i3 = indices[j + n + n];
    atomicAdd(grad_x + l * m + i1, grad_y[i] * w1);
    atomicAdd(grad_x + l * m + i2, grad_y[i] * w2);
    atomicAdd(grad_x + l * m + i3, grad_y[i] * w3);
  }
}

void three_nearest_neighbors_interpolate_grad(int b, int c, int n, int m,
                                              const float *grad_y,
                                              const int *indices,
                                              const float *weights,
                                              float *grad_x) {
  three_nearest_neighbors_interpolate_grad_kernel<<<
      b, optimal_block_config(n, c), 0, at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, m, grad_y, indices, weights, grad_x);
  CUDA_CHECK_ERRORS();
}
