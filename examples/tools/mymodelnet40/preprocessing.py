import h5py
# from tqdm import tqdm
import argparse

import torch

from libs import tpcpath
from torchpcp.datasets.PointNet2.ModelNet import ModelNet40

def create_modelnet_dataset(file_name, dataset):
    loader = range(len(dataset))
    # loader = tqdm(loader, desc=file_name)
    with h5py.File(file_name, "w") as f:
        point_clouds = []
        labels = []
        for i in loader:
            points, label = dataset[i]
            point_clouds.append(points)
            labels.append(label)

        f.create_dataset("point_clouds", data=point_clouds, compression='gzip', compression_opts=1, dtype="float32")
        f.create_dataset("labels", data=labels, compression='gzip', compression_opts=1, dtype="uint8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-p', type=str, required=True)
    parser.add_argument('--output_path', '-o', type=str, required=True)
    args = parser.parse_args()
    path = args.dataset_path
    datasets = ModelNet40(path)

    output_path = args.output_path
    preprocessed_dataset = "mymodelnet40"
    preprocessed_dataset_path = os.path.join(output_path, preprocessed_dataset)
    os.mkdir(preprocessed_dataset_path)
    train_dataset_path = os.path.join(preprocessed_dataset_path, "train_modelnet40.h5")
    create_modelnet_dataset(train_dataset_path, datasets["train"])
    test_dataset_path = os.path.join(preprocessed_dataset_path, "test_modelnet40.h5")
    create_modelnet_dataset(test_dataset_path, datasets["test"])

if __name__ == "__main__":
    main()
