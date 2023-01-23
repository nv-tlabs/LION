#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    coords : coords of each point, IntTensor[b, 3, n]
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
*/
__global__ void grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const int *__restrict__ coords,
                                  int *__restrict__ ind, int *cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  ind += batch_index * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    // if (ind[i] == -1)
    //   continue;
    ind[i] = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
    atomicAdd(cnt + ind[i], 1);
  }
}

/*
  Function: average pool voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void avg_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ feat,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(out + j * s + pos, feat[j * n + i] * div_cur_cnt);
      }
    }
  }
}

/*
  Function: average pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void avg_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ ind,
                                         const int *__restrict__ cnt,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  cnt += batch_index * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * div_cur_cnt);
      }
    }
  }
}

void avg_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *cnt, float *out) {
  grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, coords, ind,
                                                   cnt);
  avg_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, ind, cnt,
                                                     feat, out);
  CUDA_CHECK_ERRORS();
}

void avg_voxelize_grad(int b, int c, int n, int s, const int *ind,
                       const int *cnt, const float *grad_y, float *grad_x) {
  avg_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, ind, cnt,
                                                          grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
