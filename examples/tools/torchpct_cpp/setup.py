import sys
sys.path.append("../../../")  # for torch_point_cloud in this repository

from setuptools import setup, find_packages
from torch_point_cloud.modules.functional.torchpct_cpp.backend import sources
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = 'torchpct_cpp'
VERSION = '0.1.0'
DESCRIPTION = 'Point cloud processing tools'
URL = 'https://github.com/Obarads/torch_point_cloud'
AUTHOR = 'Obarads'
LICENSE = 'MIT License'

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
                PACKAGE_NAME,
                sources=sources,
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                }
            )
        ],
        cmdclass={'build_ext': BuildExtension}
    )

if __name__ == "__main__":
    main()