import os
import h5py
import glob
import numpy as np

from torch_point_cloud.datasets.utils.common import download_and_unzip

# from PointNet
# https://github.com/charlesq34/pointnet/blob/539db60eb63335ae00fe0da0c8e38c791c764d2b/provider.py#L90
def load_file(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
    return (data, label)

def load_files(h5_filenames):
    all_data = []
    all_label = []
    for h5_name in h5_filenames:
        data, label = load_file(h5_name)
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

def get_paths(root, split):
    paths = glob.glob(os.path.join(root, 'ply_data_%s*.h5'%split))
    return paths

# https://github.com/charlesq34/pointnet/blob/master/provider.py
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
