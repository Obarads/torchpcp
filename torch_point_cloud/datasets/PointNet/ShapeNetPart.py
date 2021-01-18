import os

import numpy as np

import torch
from torch.utils.data import Dataset

from .utils import provider
from torch_point_cloud.utils.setting import (
    make_folders, download_and_unzip)

class ShapeNetPartDataset(Dataset):
    def __init__(self, root, num_points, split):
        assert split in ["train", "test", "val"]

        if split == "train":
            filename = "train_hdf5_file_list.txt"
        elif split == "test":
            filename = "test_hdf5_file_list.txt"
        elif split == "val":
            filename = "val_hdf5_file_list.txt"

        file_list = provider.getDataFiles(os.path.join(root, filename))

        point_clouds = []
        cla_labels = []
        seg_labels = []
        for i in range(len(file_list)):
            file_path = os.path.join(root, file_list[i].split("/")[-1])
            file_point_clouds, file_cla_labels, file_seg_labels = \
                provider.loadDataFile_with_seg(file_path)
            file_point_clouds = file_point_clouds[:, 0:num_points, :]
            file_seg_labels = file_seg_labels[:, 0:num_points]
            point_clouds.append(file_point_clouds)
            cla_labels.append(file_cla_labels)
            seg_labels.append(file_seg_labels)

        self.point_clouds = np.concatenate(point_clouds, axis=0)
        self.cla_labels = np.concatenate(cla_labels, axis=0)
        self.seg_labels = np.concatenate(seg_labels, axis=0)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        cla_label = self.cla_labels[idx][0]
        seg_label = self.seg_labels[idx]

        return point_cloud, cla_label, seg_label

class ShapeNetPart(dict):
    def __init__(self, root, num_points, split=None):
        super().__init__()
        if split is None:
            split = ['train', 'test', 'val']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = ShapeNetPartDataset(root=root, num_points=num_points, split=s)

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/train_partseg.py
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes, device=y.device)[y.cpu().data.numpy(),]
    return new_y

# https://github.com/charlesq34/pointnet/blob/master/provider.py
def download_ShapeNetPart(path):
    """
    Download ShapeNet and Part dataset.
    Parameters
    ----------
    path: str
        path to save a dataset
    """

    # Download dataset for point cloud classification
    DATA_DIR = path
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenetcore_partanno_v0')):
        download_and_unzip("https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip", DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data')):
        download_and_unzip("https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip", DATA_DIR)

def rotation_and_jitter(batch):
    point_clouds, cla_labels, seg_labels = list(zip(*batch))

    point_clouds = np.array(point_clouds, dtype=np.float32)
    cla_labels = np.array(cla_labels, dtype=np.long)
    seg_labels = np.array(seg_labels, dtype=np.long)

    point_clouds = provider.rotate_point_cloud(point_clouds)
    point_clouds = provider.jitter_point_cloud(point_clouds)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
    cla_labels = torch.tensor(cla_labels, dtype=torch.long)
    seg_labels = torch.tensor(seg_labels, dtype=torch.long)

    return point_clouds, labels, seg_labels
