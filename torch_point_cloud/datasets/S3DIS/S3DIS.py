import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import glob
import h5py
import yaml
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset

from torch_point_cloud.datasets.ASIS.utils.provider import (
    loadDataFile_with_groupseglabel_stanfordindoor)
from torch_point_cloud.datasets.ASIS.utils.indoor3d_util import (
    room2blocks_wrapper_normalized)
from torch_point_cloud.utils.converter import (
    batch_sparseLabel_to_denseLabel, sparseLabel_to_denseLabel)

# for Preprocessing
from .utils.data_prep_util import save_h5ins
from .utils.indoor3d_util import collect_point_label

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

def get_paths(root, area_numbers, extension=".h5"):
    assert extension in [".h5", ".npy"]

    area_list = []
    for i in area_numbers:
        area_list.append(f"Area_{i}")

    h5_file_path_list = []

    # load SPG data and then convert SPG to graphs.
    for area in area_list:
        paths = glob.glob(os.path.join(root, area+"*"+extension))
        for h5_file_path in sorted(paths):
            if h5_file_path.endswith(extension):
                h5_file_path_list.append(h5_file_path)

    return h5_file_path_list

class S3DISSceneDataset(Dataset):
    def __init__(self, dataset_path, split, test_area=5):
        """
        Get S3DIS dataset with scenes.

        Parameters
        ----------
        dataset_path : str
            Path of S3DIS dataset created by Preprocessing.create_scene_dataset.
        split : {'train', 'test'}
            Purpose of data. 
        test_area : {1, 2, 3, 4, 5, 6}
            Test area number.
        """
        # Create area number list.
        if split == "train":
            area_numbers = list(range(1,7))
            area_numbers.pop(test_area - 1)
        elif split == "test":
            area_numbers = [test_area]

        # Get data file path.
        scene_path_list = get_paths(dataset_path, area_numbers, extension=".npy")

        # args
        self.dataset_path = dataset_path
        self.split = split
        self.test_area = test_area

        # other params
        self.num_classes = 13
        self.scene_path_list = scene_path_list

    def __len__(self):
        # Return number of rooms.
        return len(self.scene_path_list)

    def __getitem__(self, idx):
        """
        Get a scene point cloud, instance label and semantic label.

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
        scene_path : str
            A scene path.
        """
        # get a room path
        scene_path = self.scene_path_list[idx]
        # load a scene data
        scene_data = np.load(scene_path)

        # get scene data
        point_cloud = scene_data[:, 0:6] # coords and RGB(0~255), xyzrgb
        sem_labels = scene_data[:, 6] # semantic labels per point
        ins_labels = scene_data[:, 7] # instance labels per point

        return point_cloud, sem_labels, ins_labels, scene_path

class S3DISBlockDataset(Dataset):
    def __init__(self, dataset_path, num_points, split, test_area=5):
        """
        Get S3DIS dataset with blocks.

        Parameters
        ----------
        dataset_path : str
            Path of S3DIS dataset created by Preprocessing.create_block_dataset.
        num_points : int
            Number of points in each block.
        split : {'train', 'test'}
            Purpose of data. 
        test_area : {1, 2, 3, 4, 5, 6}
            Test area number.
        """
        # Create area number list.
        if split == "train":
            area_numbers = list(range(1,7))
            area_numbers.pop(test_area - 1)
        elif split == "test":
            area_numbers = [test_area]

        # Get data file path.
        block_path_list = get_paths(dataset_path, area_numbers, extension=".h5")

        # create a list to get a scene name and block index from dataset index
        idx_to_datainfo = []
        for filename in block_path_list:
            with h5py.File(filename, "r") as f:
                ins_label = f["pid"]
                num_blocks = len(ins_label)
            for i in range(num_blocks):
                idx_to_datainfo.append([filename,i])

        # args
        self.dataset_path = dataset_path
        self.num_points = num_points
        self.split = split
        self.test_area = test_area

        # other params
        self.num_classes = 13
        self.scene_path_list = block_path_list
        self.idx_to_datainfo = idx_to_datainfo

    def __len__(self):
        # Return number of all blocks.
        return len(self.idx_to_datainfo)

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
        datainfo : [str, str]
            A scene path and block index.
        """

        # from dataset index to a scene name and block index
        scene_path, block_idx = self.idx_to_datainfo[idx]

        # load a block file
        with h5py.File(scene_path, 'r') as f:
            # from file data to numpy format
            # coordinates (xyz) and colors (RGB)
            points = f["data"][block_idx][0:self.num_points].astype(np.float32)
            # instance labels per point
            ins_label = f["pid"][block_idx][0:self.num_points]
            ins_label = np.array(sparseLabel_to_denseLabel(ins_label), dtype=np.int32)
            # semantic labels per point
            if 'seglabel' in f:
                sem_label = f['seglabel']
            else:
                sem_label = f['seglabels']
            sem_label = sem_label[block_idx][0:self.num_points].astype(np.int32)

        # create datainfo
        datainfo = [scene_path, block_idx]

        return points, sem_label, ins_label, datainfo

