"""
Base: https://github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py
Access: 2020/8/1
"""

import os
import pickle

import numpy as np

from torch.utils.data import Dataset

from torch_point_cloud.utils.converter import write_ply

__all__ = ["ScanNet"]

class _ScanNetDataset(Dataset):
    def __init__(self, root, num_points=8192, split='train'):
        self.npoints = num_points
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))

        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding="bytes")
            self.semantic_labels_list = pickle.load(fp, encoding="bytes")

        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)

    def __len__(self):
        return len(self.scene_points_list)

    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        # point_set = np.array(point_set,np.float32)
        # write_ply("scannet.ply",point_set)
        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)
        smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:]
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg)==0:
                continue
            mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        # sample_weight = self.labelweights[semantic_seg]
        # sample_weight *= mask
        # return point_set, semantic_seg, sample_weight
        return point_set, semantic_seg, mask

class ScanNet(dict):
    def __init__(self, root, num_points, split=None):
        super().__init__()
        if split is None:
            split = ['train', 'test']
        elif not isinstance(split, (list, tuple)):
            split = [split]
        for s in split:
            self[s] = _ScanNetDataset(root=root, num_points=num_points, split=s)

