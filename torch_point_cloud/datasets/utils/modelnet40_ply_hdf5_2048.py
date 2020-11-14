import os
import h5py
import glob
import numpy as np

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


