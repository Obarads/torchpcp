import random

import torch
from torch import nn

from torch_point_cloud.modules.XConv import XConv
from torch_point_cloud.modules.Layer import LinearModule
from torch_point_cloud.modules.functional import sampling

class PointCNNClassification(nn.Module):
    def __init__(self, point_feature_size, out_channel, use_x_transform=True):
        super().__init__()

        self.xconv1 = XConv(
            in_channel=3, 
            out_channel=48, 
            k=8, 
            dilation=1, 
            depth_multiplier=4
        )
        self.xconv2 = XConv(
            in_channel=48,
            out_channel=96,
            k=12,
            dilation=2,
            depth_multiplier=2
        )
        self.xconv3 = XConv(
            in_channel=96,
            out_channel=192,
            k=16,
            dilation=2,
            depth_multiplier=2
        )
        self.xconv4 = XConv(
            in_channel=192,
            out_channel=384,
            k=16,
            dilation=3,
            depth_multiplier=2
        )

        self.fc =  nn.Sequential(
            LinearModule(384, 384),
            LinearModule(384, 192),
            nn.Dropout(0.2),
            nn.Linear(192, out_channel)
        )

        self.point_feature_size = point_feature_size

    def forward(self, inputs):
        # B, C, N = inputs.shape
        if self.point_feature_size > 0:
            coords = inputs[:, :3] # xyz coords
            features = inputs[:, 3:] # features other than coords
        else:
            coords = inputs
            features = None
        
        # get center coords
        center_coords_1 = coords
        # XConv
        features = self.xconv1(center_coords_1, coords, features)

        # subsampling
        center_coords_2 = self.subsampling(center_coords_1, 384)
        # XConv
        features = self.xconv2(center_coords_2, center_coords_1, features)

        # subsampling
        center_coords_3 = self.subsampling(center_coords_2, 128)
        # XConv
        features = self.xconv3(center_coords_3, center_coords_2, features)

        # subsampling
        center_coords_4 = self.subsampling(center_coords_3, 128)
        # XConv
        features = self.xconv4(center_coords_4, center_coords_3, features)

        res = self.fc(features)

        return res

    def subsampling(self, coords, num_samples):
        N = coords.shape[2]
        sampled_point_indices = sampling.random_sampling(N, num_samples)
        center_coords = sampling.index2points(coords, sampled_point_indices)
        return center_coords


