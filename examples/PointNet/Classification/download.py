import os, sys
sys.path.append("./") # for package path
sys.path.append("../../../") # for torch_point_cloud in this repository

import argparse

from torch_point_cloud.datasets.PointNet.ModelNet import download_ModelNet40

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()

    download_ModelNet40(args.path)


