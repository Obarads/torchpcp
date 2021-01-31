import os, sys
CW_DIR = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import argparse

from torchpcp.datasets.ASIS.S3DIS import Preprocessing
from torchpcp.utils.setting import make_folders

def main():
    parser = argparse.ArgumentParser(description="create a dataset with h5")
    parser.add_argument("--dataset_root", "-d", type=str, required=True,
                        help="""path to S3DIS dataset (S3DIS dataset: 
                        Stanford3dDataset_v1.2_Aligned_Version.zip)""")
    parser.add_argument("--output_root", "-o", type=str, required=True, 
                        help="""path to a folder of preprocessed data 
                        (outputs)""")
    parser.add_argument("--create_blocks", "-c", type=bool, default=True,
                        help="""If this argument is set to true, the processing 
                        speed of deep model increases but the storage is 
                        occupied.""")
    parser.add_argument("--block_size", type=float, default=1.0, 
                        help="""Block size is this args, if --create_blocks 
                        is true.""")
    parser.add_argument("--stride", type=float, default=0.5,
                        help="""Stride is this args, if --create_blocks 
                        is true.""")
    parser.add_argument("--num_points", type=float, default=4096,
                    help="""Preprocessing to create blocks samples `num_points` 
                    points if --create_blocks is true.""")
    cfg = parser.parse_args()

    scene_output_root = os.path.join(cfg.output_root, "scenes")
    make_folders(scene_output_root)
    print("create a scene dataset")
    Preprocessing.create_scene_dataset(cfg.dataset_root, scene_output_root)

    if cfg.create_blocks:
        block_output_root = os.path.join(cfg.output_root, "blocks")
        make_folders(block_output_root)
        print("create a block dataset")
        Preprocessing.create_block_dataset(scene_output_root, block_output_root, 
                                           cfg.num_points, cfg.block_size, 
                                           cfg.stride)

if __name__ == "__main__":
    main()
