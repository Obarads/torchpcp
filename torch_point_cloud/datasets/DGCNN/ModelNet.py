import os
import h5py
import glob
import numpy as np

from torch.utils.data import Dataset

from torch_point_cloud.datasets.utils import modelnet40_ply_hdf5_2048

# https://github.com/WangYueFt/dgcnn/blob/e96a7e26555c3212dbc5df8d8875f07228e1ccc2/pytorch/data.py#L64
class ModelNet40Path2Object(Dataset):
    def __init__(self, file_paths, num_points, split):
        data, label = modelnet40_ply_hdf5_2048.load_files(file_paths)
        self.data = data
        self.label = np.squeeze(label, axis=1)
        
        self.num_points = num_points
        self.split = split
        self.num_classes = 40

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.split == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ModelNet40(dict):
    def __init__(self, dataset_root, num_points=1024, split=["train","test"]):
        # assert isinstance(area_dict, dict)
        for key in split:
            path_list = modelnet40_ply_hdf5_2048.get_paths(dataset_root, key)
            self[key] = ModelNet40Path2Object(path_list, num_points, split)

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


