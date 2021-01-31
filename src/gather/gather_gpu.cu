#include "../cuda_utils.cuh"

__global__ void gather_kernel(
    const int b, const int c, const int n,
    const int point_idx_size,
    const float *__restrict__ point_clouds,
    const int *__restrict__ indices,
    float *__restrict__ outputs)
{
    // int batch_index = blockIdx.x;
    // int index = threadIdx.x;
    // int stride = blockDim.x;

    // point_clouds += batch_index *

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= point_idx_size) return;

    outputs += bs_idx * c * point_idx_size + c_idx * point_idx_size + pt_idx;
    indices += bs_idx * point_idx_size + pt_idx;
    point_clouds += bs_idx * c * n + c_idx * n;
    outputs[0] = point_clouds[indices[0]];

}

void gather(
    const int b, const int c, const int n,
    const int point_idx_size,
    const float *point_clouds,
    const int *indices,
    float *outputs)
{
    gather_kernel<<<
        b, optimal_num_threads(point_idx_size), 0,
        at::cuda::getCurrentCUDAStream()>>>(
        b, c, n,
        point_idx_size,
        point_clouds,
        indices,
        outputs);
    CUDA_CHECK_ERRORS();
}
