import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import glob
import h5py
import yaml
import tqdm
import numpy as np

from torch.utils.data import Dataset

from torch_point_cloud.datasets.ASIS.utils.provider import (
    loadDataFile_with_groupseglabel_stanfordindoor)
from torch_point_cloud.datasets.ASIS.utils.indoor3d_util import (
    room2blocks_wrapper_normalized)
from torch_point_cloud.utils.converter import (
    batch_sparseLabel_to_denseLabel, sparseLabel_to_denseLabel)

def create_batch_instance_information(batch_ins_labels):
    """
    create new instance labels, masks, number of instances list and maximum number of instance. 
    Parameters
    ----------
    ins_labels:np.array (shape:(B, N))
        instance labels

    Returns
    -------
    ins_labels: np.array (shape: (B, N))
        modified instance labels
    ins_masks: np.array (shape: (B, N, C))
        instance masks
    ins_label_sizes: np.array (shape: (B))
        instance size
    max_ins_label: int
        max instance size

    Note
    ----
    B:
        Batch size
    N:
        Number of points
    C:
        Max instance size
    """
    ins_labels = batch_ins_labels
    ins_masks = []
    ins_label_sizes = []

    # modify instance labels
    ins_labels = batch_sparseLabel_to_denseLabel(ins_labels)

    # get max instance label for instance mask
    max_ins_label = np.amax(ins_labels) + 1
    num_points = ins_labels.shape[1]

    # create ins masks and ins size
    for ins_label in ins_labels:
        ins_label_sizes.append(np.unique(ins_label).size)
        ins_mask = np.zeros((num_points, max_ins_label), 
                             dtype=np.float32)
        ins_mask[np.arange(num_points), ins_label] = 1
        ins_masks.append(ins_mask)

    return ins_labels, ins_masks, ins_label_sizes, max_ins_label

class S3DISPath2SceneDataset(Dataset):
    def __init__(self, path_list, num_points, block_size=1.0, stride=0.5):
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride

        self.ROOM_PATH_LIST = path_list

    def __len__(self):
        return len(self.ROOM_PATH_LIST)

    def __getitem__(self, idx):
        room_path = self.ROOM_PATH_LIST[idx]
        cur_data, cur_sem, cur_group = room2blocks_wrapper_normalized(
            room_path, self.num_points, block_size=self.block_size, 
            stride=self.stride, random_sample=False, sample_num=None)
        cur_data = cur_data[:, 0:self.num_points, :]
        cur_sem = np.squeeze(cur_sem)
        cur_group = np.squeeze(cur_group)
        room_name = room_path.split("/")[-1][:-4]

        # get scene data
        raw_scene_data = np.load(room_path)

        return cur_data, cur_sem, cur_group, room_name, raw_scene_data

class S3DISPath2BlockDataset(Dataset):
    def __init__(self, path_list, num_points):

        self.num_points = num_points
        self.path_list = path_list

        point_clouds = []
        ins_labels = []
        sem_labels = []

        for h5_file_path in path_list:
            cur_data, cur_group, _, cur_sem = \
                loadDataFile_with_groupseglabel_stanfordindoor(h5_file_path)
            # add point clouds to list
            point_clouds.append(cur_data[:, 0:self.num_points, :])
            # convert ins labels in a scene to each block
            cur_group = batch_sparseLabel_to_denseLabel(
                cur_group[:, 0:self.num_points])
            # add ins labels to list
            ins_labels.append(cur_group)
            # add sem labels to list
            sem_labels.append(cur_sem[:, 0:self.num_points])

        # save each data
        self.point_clouds = np.concatenate(point_clouds, axis=0)
        self.ins_labels = np.concatenate(ins_labels, axis=0)
        self.sem_labels = np.concatenate(sem_labels, axis=0)

        ins_labels, ins_masks, ins_label_sizes, max_ins_label \
            = create_batch_instance_information(self.ins_labels)

        self.ins_labels = ins_labels
        self.ins_masks = ins_masks
        self.ins_label_sizes = ins_label_sizes
        self.max_ins_label = max_ins_label

        _, weights = np.unique(self.sem_labels, return_counts=True)
        weights = 1 / weights
        self.weights = weights / np.sum(weights)
        # self.weights = 1/np.log(1.2+weights)

    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        points = self.point_clouds[idx]
        ins_label = self.ins_labels[idx]
        sem_label = self.sem_labels[idx]
        ins_label_size = self.ins_label_sizes[idx]
        ins_mask = self.ins_masks[idx]

        return points, sem_label, ins_label, ins_label_size, ins_mask

class S3DISPath2BlockDatasetMS(Dataset):
    """
    MS=memory saving
    """
    def __init__(self, path_list, num_points):
        self.num_points = num_points
        self.path_list = path_list

        # define each data
        idx2datainfo = []
        max_ins_label_size = 0

        for h5_filename in path_list:
            f = h5py.File(h5_filename, "r")
            ins_label = f["pid"]
            ins_label_size = np.amax(ins_label)
            if ins_label_size > max_ins_label_size:
                max_ins_label_size = ins_label_size
            num_blocks = len(ins_label)
            for i in range(num_blocks):
                idx2datainfo.append([h5_filename,i])

        self.idx2datainfo = idx2datainfo
        self.max_ins_label_size = max_ins_label_size

    def __len__(self):
        return len(self.idx2datainfo)

    def __getitem__(self, idx):
        scene_name, block_idx = self.idx2datainfo[idx]
        f = h5py.File(scene_name, "r")

        points = f["data"][block_idx][0:self.num_points].astype(np.float32)
        ins_label = f["pid"][block_idx][0:self.num_points]
        ins_label = np.array(sparseLabel_to_denseLabel(ins_label), dtype=np.int32)
        if 'seglabel' in f:
            sem_label = f['seglabel']
        else:
            sem_label = f['seglabels']
        sem_label = sem_label[block_idx][0:self.num_points].astype(np.int32)

        ins_mask = np.zeros((self.num_points, self.max_ins_label_size), dtype=np.float32)
        ins_mask[np.arange(self.num_points), ins_label] = 1
        ins_label_size = np.unique(ins_label).size

        return points, sem_label, ins_label, ins_label_size, ins_mask

class Scene2Blocks(Dataset):
    """
    scene: single scene
    この機能、tntにあったような
    """
    def __init__(self, scene, num_points):
        # self.scene_dataset = scene
        self.point_clouds = scene[0][:, 0:num_points]
        self.sem_labels = scene[1][:, 0:num_points]
        self.scene_name = scene[3]
        self.raw_scene_data = scene[4]

        ins_labels, ins_masks, ins_label_sizes, max_ins_label = \
            create_batch_instance_information(scene[2][:, 0:num_points])

        self.ins_labels = ins_labels
        self.ins_masks = ins_masks
        self.ins_label_size = ins_label_sizes
        self.max_ins_label = max_ins_label

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        block_point_cloud = self.point_clouds[idx]
        block_sem_label = self.sem_labels[idx]
        block_ins_label = self.ins_labels[idx]
        block_ins_mask = self.ins_masks[idx]
        block_ins_label_size = self.ins_label_size[idx]

        # scene_name = self.scene_name
        # raw_scene_data = self.raw_scene_data
        return block_point_cloud, block_sem_label, block_ins_label, \
            block_ins_mask, block_ins_label_size

class S3DISSceneDataset(dict):
    def __init__(self, root, num_points, area_dict, block_size=1.0, stride=0.5):
        super().__init__()
        # assert isinstance(area_dict, dict)
        for key in area_dict:
            path_list = get_block_paths(root, area_dict[key], ".npy")
            self[key] = S3DISPath2SceneDataset(path_list, num_points, 
                                               block_size=block_size,
                                               stride=stride)

class S3DISBlockDataset(dict):
    def __init__(self, root, num_points, area_dict, memory_saving=False):
        super().__init__()
        # assert isinstance(area_dict, dict)
        for key in area_dict:
            path_list = get_block_paths(root, area_dict[key], ".h5")
            if memory_saving:
                self[key] = S3DISPath2BlockDatasetMS(path_list, num_points)
            else:
                self[key] = S3DISPath2BlockDataset(path_list, num_points)

def get_block_paths(root, areas, extension):
    assert extension in [".h5", ".npy"]

    area_list = []
    for i in areas:
        area_list.append(f"Area_{i}")

    h5_file_path_list = []

    # load SPG data and then convert SPG to graphs.
    for area in area_list:
        paths = glob.glob(os.path.join(root, area+"*"+extension))
        for h5_file_path in sorted(paths):
            if h5_file_path.endswith(extension):
                h5_file_path_list.append(h5_file_path)

    return h5_file_path_list
