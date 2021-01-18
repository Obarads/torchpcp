import os

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
sources=[
    os.path.join(_src_path,'src', f) for f in [
        'ball_query/ball_query.cpp',
        'ball_query/ball_query.cu',
        'grouping/grouping.cpp',
        'grouping/grouping.cu',
        'interpolate/neighbor_interpolate.cpp',
        'interpolate/neighbor_interpolate.cu',
        'interpolate/trilinear_devox.cpp',
        'interpolate/trilinear_devox.cu',
        'sampling/sampling.cpp',
        'sampling/sampling.cu',
        'voxelization/vox.cpp',
        'voxelization/vox.cu',
        'knn/k_nearest_neighbors.cpp',
        'knn/k_nearest_neighbors.cu',
        "gather/gather.cpp",
        "gather/gather.cu",
        'bindings.cpp',
    ]
]

_backend = load(name='_torch_c',
                extra_cflags=['-O3', '-std=c++17'],
                sources=sources
            )

__all__ = ['_backend', 'sources']
