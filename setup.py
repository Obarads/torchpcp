from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PACKAGE_NAME = 'torchpcp'
VERSION = '0.1.0'
DESCRIPTION = 'Point cloud processing with deep learning'
URL = 'https://github.com/Obarads/torchpcp'
AUTHOR = 'Obarads'
LICENSE = 'MIT License'

import sys, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sources = [
    os.path.join(BASE_DIR, 'src', f) for f in [
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
        # "gather/gather.cpp",
        # "gather/gather_gpu.cu",
        'bindings.cpp',
    ]
]

def main():
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        url=URL,
        license=LICENSE,
        packages=find_packages(),
        ext_modules=[
            CUDAExtension(
                "{}.modules.functional.backend.cpp_ex".format(PACKAGE_NAME),
                sources=sources,
            )
        ],
        cmdclass={'build_ext': BuildExtension}
    )

if __name__ == "__main__":
    main()
