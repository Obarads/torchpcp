import os
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

from tqdm import tqdm

from .utils import provider

class ModelNet40(dict):
    def __init__(self, root, split=None):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = ModelNet40Dataset(root=root, split=s)

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNet40Dataset(Dataset): # modelnet40_normal_resampled
    def __init__(self, root, split, with_fastlist=True):
        modelnet10 = False

        self.root = root
        self.num_points = 1024
        self.normalize = True
        cache_size=25000

        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        self.normal_channel = True
        
        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))] 
            shape_ids['test']= [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert(split=='train' or split=='test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i])+'.txt') for i in range(len(shape_ids[split]))]

        if with_fastlist:
            fastlist = []
            loader = tqdm(range(len(self.datapath)), desc="Load modelnet40 {} dataset.".format(split))
            for idx in loader:
                fn = self.datapath[idx]
                cls = self.classes[self.datapath[idx][0]]
                cls = np.squeeze(np.array([cls]).astype(np.int32))
                point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
                # Take the first npoints
                point_set = point_set[0:self.num_points,:]
                if self.normalize:
                    point_set[:,0:3] = pc_normalize(point_set[:,0:3])
                if not self.normal_channel:
                    point_set = point_set[:,0:3]
                fastlist.append([point_set, cls])
            self.fastlist = fastlist
            self.getitme_function = self._getitem_fastlist
        else:
            self.cache_size = cache_size # how many data points to cache in memory
            self.cache = {} # from index to (point_set, cls) tuple
            self.getitme_function = self._getitem_cache

    def __len__(self):
        return len(self.datapath)

    def _getitem_fastlist(self, idx):
        return self.fastlist[idx]
    
    def _getitem_cache(self, idx):
        if idx in self.cache:
            point_set, cls = self.cache[idx]
        else:
            fn = self.datapath[idx]
            cls = self.classes[self.datapath[idx][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1],delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set = point_set[0:self.num_points,:]
            if self.normalize:
                point_set[:,0:3] = pc_normalize(point_set[:,0:3])
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                self.cache[idx] = (point_set, cls)
        return point_set, cls

    def __getitem__(self, idx):
        return self.getitme_function(idx)

    def augment_batch_data(self, batch):
        point_clouds, labels = list(zip(*batch))

        point_clouds = np.array(point_clouds, dtype=np.float32)
        labels = np.array(labels, dtype=np.long)

        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(point_clouds)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(point_clouds)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data

        rotated_data = torch.tensor(rotated_data)
        labels = torch.tensor(labels)

        return provider.shuffle_points(rotated_data), labels

def rotation_and_jitter(batch):
    point_clouds, labels = list(zip(*batch))

    point_clouds = np.array(point_clouds, dtype=np.float32)
    labels = np.array(labels, dtype=np.long)

    point_clouds = provider.rotate_point_cloud(point_clouds)
    point_clouds = provider.jitter_point_cloud(point_clouds)

    point_clouds = torch.tensor(point_clouds, dtype=torch.float32)
    labels = torch.tensor(labels)

    return point_clouds, labels

class MyModelNet40Dataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.normal_channel = True
        if split == "train":
            path = os.path.join(self.root, "train_modelnet40.h5")
        elif split == "test":
            path = os.path.join(self.root, "test_modelnet40.h5")
        else:
            raise NotImplementedError()
            
        with h5py.File(path, "r") as f:
            self.point_clouds = f['point_clouds'][:]
            self.labels = f['labels'][:]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]
        return point_cloud, label
    
    def augment_batch_data(self, batch):
        point_clouds, labels = list(zip(*batch))

        point_clouds = np.array(point_clouds, dtype=np.float32)
        labels = np.array(labels, dtype=np.long)

        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(point_clouds)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(point_clouds)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

        jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = provider.shift_point_cloud(jittered_data)
        jittered_data = provider.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data

        rotated_data = torch.tensor(rotated_data)
        labels = torch.tensor(labels)

        return provider.shuffle_points(rotated_data), labels

class MyModelNet40(dict):
    def __init__(self, root, split=None):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = MyModelNet40Dataset(root=root, split=s)

