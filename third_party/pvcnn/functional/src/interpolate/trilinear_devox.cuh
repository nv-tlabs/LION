#ifndef _TRILINEAR_DEVOX_CUH
#define _TRILINEAR_DEVOX_CUH

// CUDA function declarations
void trilinear_devoxelize(int b, int c, int n, int r, int r2, int r3,
                          bool is_training, const float *coords,
                          const float *feat, int *inds, float *wgts,
                          float *outs);
void trilinear_devoxelize_grad(int b, int c, int n, int r3, const int *inds,
                               const float *wgts, const float *grad_y,
                               float *grad_x);

#endif
