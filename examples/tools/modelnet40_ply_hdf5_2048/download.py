import os, sys
sys.path.append("../../../") # for torchpcp in this repository

import argparse

from torchpcp.datasets.utils.modelnet40_ply_hdf5_2048 import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True)
    args = parser.parse_args()

    download(args.path)


