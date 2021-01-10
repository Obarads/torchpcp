import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELNET40_DATAPATH = os.path.abspath(os.path.join(BASE_DIR, "../data/ply_data_train0.h5"))
S3DIS_DATAPATH = os.path.abspath(os.path.join(BASE_DIR, "../data/Area_1_office_1.h5"))

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from torch_point_cloud.datasets.ModelNet.utils import provider
from torch_point_cloud.utils.converter import sparseLabel_to_denseLabel

class SimpleObjectDataset(Dataset):
    def __init__(self, num_points=1024, file_path=MODELNET40_DATAPATH):
        point_clouds = []
        labels = []

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

class SimpleSceneDataset(Dataset):
    def __init__(self, num_points=4096, file_path=S3DIS_DATAPATH):
        idx_to_datainfo = []
        with h5py.File(file_path, "r") as f:
            points = f["data"][:, :num_points].astype(np.float32)
            # instance labels per point
            ins_label = f["pid"][:, :num_points].astype(np.int32)
            ins_label = np.array(sparseLabel_to_denseLabel(ins_label), dtype=np.int32)
            # semantic labels per point
            if 'seglabel' in f:
                sem_label = f['seglabel']
            else:
                sem_label = f['seglabels']
            sem_label = sem_label[:, :num_points].astype(np.int32)

        self.point_clouds = points
        self.ins_labels = ins_label
        self.sem_labels = sem_label
        self.num_points = num_points

    def __len__(self):
        # Return number of all blocks.
        return len(self.point_clouds)

    def __getitem__(self, idx):
        """
        Get a block point cloud, instance label and semantic label.

        Parameters
        ------
        idx : int
            a index of dataset.
        
        Returns
        -------
        point_cloud : numpy.ndarray
            Coordinates (x,y,z) and colors (R,G,B).
        sem_labels : numpy.ndarry
            Semantic labels per point.
        ins_labels : numpy.ndarry
            Instance labels per point.
        """

        points = self.point_clouds[idx]
        sem_label = self.sem_labels[idx]
        ins_label = self.ins_labels[idx]

        return points, sem_label, ins_label

