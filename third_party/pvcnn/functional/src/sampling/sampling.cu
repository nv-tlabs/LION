#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: gather centers' features (forward)
  Args:
    b   : batch size
    c   : #channles of features
    n   : number of points in point clouds
    m   : number of query/sampled centers
    features: points' features, FloatTensor[b, c, n]
    indices : centers' indices in points, IntTensor[b, m]
    out     : gathered features, FloatTensor[b, c, m]
*/
__global__ void gather_features_kernel(int b, int c, int n, int m,
                                       const float *__restrict__ features,
                                       const int *__restrict__ indices,
                                       float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int channel_index = blockIdx.y;
  int temp_index = batch_index * c + channel_index;
  features += temp_index * n;
  indices += batch_index * m;
  out += temp_index * m;

  for (int j = threadIdx.x; j < m; j += blockDim.x) {
    out[j] = features[indices[j]];
  }
}

void gather_features(int b, int c, int n, int m, const float *features,
                     const int *indices, float *out) {
  gather_features_kernel<<<dim3(b, c, 1), optimal_num_threads(m), 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, m, features, indices, out);
  CUDA_CHECK_ERRORS();
}

/*
  Function: gather centers' features (backward)
  Args:
    b   : batch size
    c   : #channles of features
    n   : number of points in point clouds
    m   : number of query/sampled centers
    grad_y  : grad of gathered features, FloatTensor[b, c, m]
    indices : centers' indices in points, IntTensor[b, m]
    grad_x  : grad of points' features, FloatTensor[b, c, n]
*/
__global__ void gather_features_grad_kernel(int b, int c, int n, int m,
                                            const float *__restrict__ grad_y,
                                            const int *__restrict__ indices,
                                            float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int channel_index = blockIdx.y;
  int temp_index = batch_index * c + channel_index;
  grad_y += temp_index * m;
  indices += batch_index * m;
  grad_x += temp_index * n;

  for (int j = threadIdx.x; j < m; j += blockDim.x) {
    atomicAdd(grad_x + indices[j], grad_y[j]);
  }
}

void gather_features_grad(int b, int c, int n, int m, const float *grad_y,
                          const int *indices, float *grad_x) {
  gather_features_grad_kernel<<<dim3(b, c, 1), optimal_num_threads(m), 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      b, c, n, m, grad_y, indices, grad_x);
  CUDA_CHECK_ERRORS();
}

/*
  Function: furthest point sampling
  Args:
    b   : batch size
    n   : number of points in point clouds
    m   : number of query/sampled centers
    coords    : points' coords, FloatTensor[b, 3, n]
    distances : minimum distance of a point to the set, IntTensor[b, n]
    indices   : sampled centers' indices in points, IntTensor[b, m]
*/
__global__ void furthest_point_sampling_kernel(int b, int n, int m,
                                               const float *__restrict__ coords,
                                               float *__restrict__ distances,
                                               int *__restrict__ indices) {
  if (m <= 0)
    return;
  int batch_index = blockIdx.x;
  coords += batch_index * n * 3;
  distances += batch_index * n;
  indices += batch_index * m;

  const int BlockSize = 512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize = 3072;
  __shared__ float buf[BufferSize * 3];

  int old = 0;
  if (threadIdx.x == 0)
    indices[0] = old;

  for (int j = threadIdx.x; j < min(BufferSize, n); j += blockDim.x) {
    buf[j] = coords[j];
    buf[j + BufferSize] = coords[j + n];
    buf[j + BufferSize + BufferSize] = coords[j + n + n];
  }
  __syncthreads();

  for (int j = 1; j < m; j++) {
    int besti = 0;   // best index
    float best = -1; // farthest distance
    // calculating the distance with the latest sampled point
    float x1 = coords[old];
    float y1 = coords[old + n];
    float z1 = coords[old + n + n];
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
      // fetch distance at block n, thread k
      float td = distances[k];
      float x2, y2, z2;
      if (k < BufferSize) {
        x2 = buf[k];
        y2 = buf[k + BufferSize];
        z2 = buf[k + BufferSize + BufferSize];
      } else {
        x2 = coords[k];
        y2 = coords[k + n];
        z2 = coords[k + n + n];
      }
      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, td);
      // update "point-to-set" distance
      if (d2 != td)
        distances[k] = d2;
      // update the farthest distance at sample step j
      if (d2 > best) {
        best = d2;
        besti = k;
      }
    }

    dists[threadIdx.x] = best;
    dists_i[threadIdx.x] = besti;
    for (int u = 0; (1 << u) < blockDim.x; u++) {
      __syncthreads();
      if (threadIdx.x < (blockDim.x >> (u + 1))) {
        int i1 = (threadIdx.x * 2) << u;
        int i2 = (threadIdx.x * 2 + 1) << u;
        if (dists[i1] < dists[i2]) {
          dists[i1] = dists[i2];
          dists_i[i1] = dists_i[i2];
        }
      }
    }
    __syncthreads();

    // finish sample step j; old is the sampled index
    old = dists_i[0];
    if (threadIdx.x == 0)
      indices[j] = old;
  }
}

void furthest_point_sampling(int b, int n, int m, const float *coords,
                             float *distances, int *indices) {
  furthest_point_sampling_kernel<<<b, 512>>>(b, n, m, coords, distances,
                                             indices);
  CUDA_CHECK_ERRORS();
}
