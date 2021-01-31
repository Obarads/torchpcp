#ifndef _GATHER_HPP
#define _GATHER_HPP

#include <torch/extension.h>

at::Tensor gather_forward(
    int point_idx_size,
    at::IntArrayRef output_shape,
    at::Tensor point_clouds, 
    at::Tensor indices
);

#endif