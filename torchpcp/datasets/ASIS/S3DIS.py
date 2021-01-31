import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import glob
import h5py
import yaml
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset

from torchpcp.datasets.ASIS.utils.provider import (
    loadDataFile_with_groupseglabel_stanfordindoor)
from torchpcp.datasets.ASIS.utils.indoor3d_util import (
    room2blocks_wrapper_normalized)
from torchpcp.utils.converter import (
    batch_sparseLabel_to_denseLabel, sparseLabel_to_denseLabel)

# for Preprocessing
from .utils.data_prep_util import save_h5ins
from .utils.indoor3d_util import collect_point_label

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
        self.num_classes = 13
        self.ROOM_PATH_LIST = path_list
        self.num_points = num_points
        self.block_size = block_size
        self.stride = stride

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

        point_clouds = []
        ins_labels = []
        sem_labels = []

        loader = tqdm(path_list, desc="loading dataset", ncols=80)
        for h5_file_path in loader:
            cur_data, cur_group, _, cur_sem = \
                loadDataFile_with_groupseglabel_stanfordindoor(h5_file_path)
            # add point clouds to list
            point_clouds.append(cur_data[:, 0:num_points, :])
            # convert ins labels in a scene to each block
            cur_group = batch_sparseLabel_to_denseLabel(
                cur_group[:, 0:num_points])
            # add ins labels to list
            ins_labels.append(cur_group)
            # add sem labels to list
            sem_labels.append(cur_sem[:, 0:num_points])

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

        self.num_classes = 13
        self.num_points = num_points
        self.path_list = path_list

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

        self.num_classes = 13
        self.num_points = num_points
        self.path_list = path_list

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

def get_block_paths(root, areas, extension=".h5"):
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

class Preprocessing:
    def __init__(self):
        print("This class is for staticmethod.")

    @staticmethod
    def create_scene_dataset(dataset_root, output_root):
        anno_relative_paths = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'utils/meta/S3DIS/anno_paths.txt'))]
        anno_absolute_paths = [os.path.join(dataset_root, p) for p in anno_relative_paths]

        g_classes = [x.rstrip() for x in open(os.path.join(BASE_DIR, 'utils/meta/S3DIS/class_names.txt'))]
        g_class2label = {cls: i for i,cls in enumerate(g_classes)}

        if not os.path.exists(output_root):
            os.mkdir(output_root)

        anno_absolute_paths = tqdm(anno_absolute_paths, ncols=65)
        # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
        for anno_path in anno_absolute_paths:
            try:
                elements = anno_path.split('/')
                out_filename = elements[-3]+'_'+elements[-2]+'.npy' # npy only
                output_path = os.path.join(output_root, out_filename)
                if os.path.exists(output_path) == False:
                    collect_point_label(anno_path, output_path, file_format='numpy')
            except:
                print(anno_path, 'ERROR!!')

    @staticmethod
    def create_block_dataset(dataset_root, output_root, num_points, block_size, 
                            stride):
        # Constants
        indoor3d_data_dir = dataset_root
        NUM_POINT = num_points
        data_dtype = 'float32'
        label_dtype = 'int32'

        # Set paths
        filelist = os.path.join(BASE_DIR, 'utils/meta/S3DIS/all_data_label.txt')
        data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for 
                            line in open(filelist)]
        output_dir = output_root
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
        fout_room = open(output_room_filelist, 'w')

        sample_cnt = 0
        for i in range(0, len(data_label_files)):
            data_label_filename = data_label_files[i]
            fname = os.path.basename(data_label_filename).strip('.npy')
            if not os.path.exists(data_label_filename):
                continue
            data, label, inslabel = room2blocks_wrapper_normalized(
                data_label_filename, NUM_POINT, block_size=block_size, 
                stride=stride, random_sample=False, sample_num=None)
            for _ in range(data.shape[0]):
                fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

            sample_cnt += data.shape[0]
            h5_filename = os.path.join(output_dir, '%s.h5' % fname)
            print('{0}: {1}, {2}, {3}'.format(h5_filename, data.shape,  
                                            label.shape, inslabel.shape))
            save_h5ins(h5_filename, data, label, inslabel, data_dtype, label_dtype)

        fout_room.close()
        print("Total samples: {0}".format(sample_cnt))



