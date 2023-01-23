#ifndef _GROUPING_CUH
#define _GROUPING_CUH

void grouping(int b, int c, int n, int m, int u, const float *features,
              const int *indices, float *out);
void grouping_grad(int b, int c, int n, int m, int u, const float *grad_y,
                   const int *indices, float *grad_x);

#endif