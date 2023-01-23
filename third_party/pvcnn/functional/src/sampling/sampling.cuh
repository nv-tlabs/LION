#ifndef _SAMPLING_CUH
#define _SAMPLING_CUH

void gather_features(int b, int c, int n, int m, const float *features,
                     const int *indices, float *out);
void gather_features_grad(int b, int c, int n, int m, const float *grad_y,
                          const int *indices, float *grad_x);
void furthest_point_sampling(int b, int n, int m, const float *coords,
                             float *distances, int *indices);

#endif
