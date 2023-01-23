#ifndef _NEIGHBOR_INTERPOLATE_CUH
#define _NEIGHBOR_INTERPOLATE_CUH

void three_nearest_neighbors_interpolate(int b, int c, int m, int n,
                                         const float *points_coords,
                                         const float *centers_coords,
                                         const float *centers_features,
                                         int *indices, float *weights,
                                         float *out);
void three_nearest_neighbors_interpolate_grad(int b, int c, int n, int m,
                                              const float *grad_y,
                                              const int *indices,
                                              const float *weights,
                                              float *grad_x);

#endif
