import os

import numpy as np
import h5py
import glob

import torch
from torch.utils.data import Dataset

from .utils import provider
from torchpcp.datasets.utils.common import download_and_unzip

class modelnet40_ply_hdf5_2048_Dataset(Dataset):
    def __init__(self, root, num_points, split):
        """
        modelnet40_ply_hdf5_2048 dataset
        """
        assert split in ["train", "test"]

        if split == "train":
            filename = "train_files.txt"
        elif split == "test":
            filename = "test_files.txt"

        
        file_list = provider.getDataFiles(os.path.join(root, filename))

        point_clouds = []
        labels = []
        for i in range(len(file_list)):
            file_path = os.path.join(root, file_list[i].split("/")[-1])
            file_point_clouds, file_labels = provider.loadDataFile(file_path)
            file_point_clouds = file_point_clouds[:, 0:num_points, :]
            point_clouds.append(file_point_clouds)
            labels.append(file_labels)
        
        self.point_clouds = np.concatenate(point_clouds, axis=0)
        self.labels = np.concatenate(labels, axis=0)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx][0]

        return point_cloud, label

class ModelNet40(dict):
    def __init__(self, root, num_points, split=None, dataset_type="modelnet40_ply_hdf5_2048"):
        super().__init__()

        # select dataset type.
        if dataset_type == "modelnet40_ply_hdf5_2048":
            ModelNet40Dataset = modelnet40_ply_hdf5_2048_Dataset
        else:
            raise NotImplementedError('Unknown dataset_type of ModelNet40: ' + dataset_type)

        # training adn test data dict
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = ModelNet40Dataset(root=root, num_points=num_points, split=s)

def rotation_and_jitter(batch):
    point_clouds, labels = list(zip(*batch))

    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.long)

    point_clouds = provider.rotate_point_cloud(point_clouds)
    point_clouds = provider.jitter_point_cloud(point_clouds)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
    labels = torch.tensor(labels)

    return point_clouds, labels

def download(path):
    """
    Download modelnet40_ply_hdf5_2048 dataset.
    Parameters
    ----------
    path: str
        path to save a dataset
    """
    # Download dataset for point cloud classification
    DATA_DIR = path
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        download_and_unzip(www, DATA_DIR)
