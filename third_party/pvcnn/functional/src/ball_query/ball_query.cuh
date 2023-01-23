#ifndef _BALL_QUERY_CUH
#define _BALL_QUERY_CUH

void ball_query(int b, int n, int m, float r2, int u,
                const float *centers_coords, const float *points_coords,
                int *neighbors_indices);

#endif
