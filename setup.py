import os
from setuptools import setup, find_packages

PACKAGE_NAME = 'torch_point_cloud'
VERSION = '0.1.0'
DESCRIPTION = 'Point cloud processing with deep learning'
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
        packages=find_packages()
    )



if __name__ == "__main__":
    main()
