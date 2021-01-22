import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
sources = [
    os.path.join(_src_path, 'src', f) for f in [
        'ball_query/ball_query.cpp',
        'ball_query/ball_query_gpu.cu',
        'grouping/grouping.cpp',
        'grouping/grouping_gpu.cu',
        'interpolate/neighbor_interpolate.cpp',
        'interpolate/neighbor_interpolate_gpu.cu',
        'interpolate/trilinear_devox.cpp',
        'interpolate/trilinear_devox_gpu.cu',
        'sampling/sampling.cpp',
        'sampling/sampling_gpu.cu',
        'voxelization/vox.cpp',
        'voxelization/vox_gpu.cu',
        'knn/k_nearest_neighbors.cpp',
        'knn/k_nearest_neighbors_gpu.cu',
        "gather/gather.cpp",
        "gather/gather_gpu.cu",
        'bindings.cpp',
    ]
]

file_exists = [os.path.exists(filepath) for filepath in sources]
if False in file_exists:
    # https://stackoverflow.com/questions/65710713/importerror-libc10-so-cannot-open-shared-object-file-no-such-file-or-director
    import torch
    import torchpct_cpp
    _backend = torchpct_cpp
else:
    # for debug
    _backend = load(name='torchpct_cpp', extra_cflags=['-O3', '-std=c++17'],
                    sources=sources)

__all__ = ['_backend', 'sources']
