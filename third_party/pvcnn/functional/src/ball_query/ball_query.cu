#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  Function: ball query
  Args:
    b   : batch size
    n   : number of points in point clouds
    m   : number of query centers
    r2  : ball query radius ** 2
    u   : maximum number of neighbors
    centers_coords: coordinates of centers, FloatTensor[b, 3, m]
    points_coords : coordinates of points, FloatTensor[b, 3, n]
    neighbors_indices : neighbor indices in points, IntTensor[b, m, u]
*/
__global__ void ball_query_kernel(int b, int n, int m, float r2, int u,
                                  const float *__restrict__ centers_coords,
                                  const float *__restrict__ points_coords,
                                  int *__restrict__ neighbors_indices) {
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;
  points_coords += batch_index * n * 3;
  centers_coords += batch_index * m * 3;
  neighbors_indices += batch_index * m * u;

  for (int j = index; j < m; j += stride) {
    float center_x = centers_coords[j];
    float center_y = centers_coords[j + m];
    float center_z = centers_coords[j + m + m];
    for (int k = 0, cnt = 0; k < n && cnt < u; ++k) {
      float dx = center_x - points_coords[k];
      float dy = center_y - points_coords[k + n];
      float dz = center_z - points_coords[k + n + n];
      float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < r2) {
        if (cnt == 0) {
          for (int v = 0; v < u; ++v) {
            neighbors_indices[j * u + v] = k;
          }
        }
        neighbors_indices[j * u + cnt] = k;
        ++cnt;
      }
    }
  }
}

void ball_query(int b, int n, int m, float r2, int u,
                const float *centers_coords, const float *points_coords,
                int *neighbors_indices) {
  ball_query_kernel<<<b, optimal_num_threads(m), 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      b, n, m, r2, u, centers_coords, points_coords, neighbors_indices);
  CUDA_CHECK_ERRORS();
}
