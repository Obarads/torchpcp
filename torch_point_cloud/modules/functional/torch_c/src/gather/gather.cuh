#ifndef _GATHER_CUH
#define _GATHER_CUH

void gather(
    const int b, const int c, const int n,
    const int point_idx_size,
    const float *point_clouds,
    const int *indices, 
    float *outputs
);

#endif

