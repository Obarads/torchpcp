#include "gather.hpp"
#include "gather_gpu.cuh"

#include "../utils.hpp"

at::Tensor gather_forward(
    int point_idx_size,
    at::IntArrayRef output_shape,
    at::Tensor point_clouds, 
    at::Tensor indices
)
{
    CHECK_CUDA(point_clouds);
    CHECK_CUDA(indices);
    CHECK_CONTIGUOUS(point_clouds);
    CHECK_CONTIGUOUS(indices);
    CHECK_IS_FLOAT(point_clouds);
    CHECK_IS_INT(indices);

    // auto tensor_sizes = indices.sizes();

    at::Tensor outputs = torch::zeros(
        output_shape, at::device(point_clouds.device()).dtype(at::ScalarType::Float)
    );

    int b = output_shape[0];
    int c = output_shape[1];
    int n = point_clouds.size(2);

    gather(
        b, c, n,
        point_idx_size,
        point_clouds.data_ptr<float>(), 
        indices.data_ptr<int>(), 
        outputs.data_ptr<float>()
    );

    return outputs;
}




