import random

import torch
from torch import nn

from torchpcp.modules.XConv import XConv
from torchpcp.modules.Layer import LinearModule, PointwiseConv2D, PointwiseConv1D
from torchpcp.modules.functional import sampling

# class Linear(nn.Module):
#     def __init__(self, in_channel, out_channel, act=nn.ReLU(), with_bn=True):
#         super().__init__()
#         self.layer = nn.Linear(in_channel, out_channel)
#         self.bn = nn.BatchNorm1d(out_channel)
#         self.act = act
#         self.with_bn = with_bn

#     def forward(self, x):
#         x = x.permute(0, 2, 1).contiguous()
#         x = self.layer(x)
#         x = x.permute(0, 2, 1).contiguous()
#         if self.with_bn:
#             x = self.bn(x)
#         if self.act is not None:
#             x = self.act(x)
#         return x

class PointCNNClassification(nn.Module):
    def __init__(self, point_feature_size, out_channel, use_x_transform=True):
        super().__init__()

        self.xconv1 = XConv(
            coord2feature_channel=48//2, # C//2
            point_feature_size=point_feature_size, # fts channels
            out_channel=48, # C
            k=8, 
            dilation=1, 
            depth_multiplier=4,
            use_x_transformation=use_x_transform
        )
        self.xconv2 = XConv(
            coord2feature_channel=48//4, # previous_C//4
            point_feature_size=48, # fts channels (previous_C)
            out_channel=96, # C
            k=12,
            dilation=2,
            depth_multiplier=2,
            use_x_transformation=use_x_transform
        )
        self.xconv3 = XConv(
            coord2feature_channel=96//4, # previous_C//4
            point_feature_size=96, # fts channels (previous_C)
            out_channel=192, # C
            k=16,
            dilation=2,
            depth_multiplier=2,
            use_x_transformation=use_x_transform
        )
        self.xconv4 = XConv(
            coord2feature_channel=192//4, # previous_C//4
            point_feature_size=192, # fts channels (previous_C)
            out_channel=384, # C
            k=16,
            dilation=3,
            depth_multiplier=2,
            use_x_transformation=use_x_transform
        )

        # self.fc =  nn.Sequential(
        #     Linear(384, 384),
        #     Linear(384, 192),
        #     nn.Dropout(0.2),
        #     nn.Linear(192, out_channel)
        # )

        # output warning (last channel warning)
        # self.fc1 = Linear(384, 384)
        # self.fc2 = Linear(384, 192)
        # self.dropout = nn.Dropout(0.2)
        # self.fc3 = Linear(192, out_channel, act=None, with_bn=False)

        self.convs = nn.Sequential(
            PointwiseConv1D(384, 384),
            PointwiseConv1D(384, 192),
            nn.Dropout(p=0.2),
            nn.Conv1d(192, out_channel, 1)
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

        # res = self.fc(features)

        # features = self.fc1(features)
        # features = self.fc2(features)
        # features = self.dropout(features)
        # res = self.fc3(features)

        res = self.convs(features)

        # res = torch.mean(features, dim=1)

        return res

    def subsampling(self, coords, num_samples):
        B, C, N = coords.shape
        sampled_point_indices = sampling.random_sampling(N, num_samples)
        sampled_point_indices = torch.tensor(sampled_point_indices, device=coords.device).view(1, num_samples).contiguous()
        sampled_point_indices = sampled_point_indices.repeat(B, 1).contiguous()
        center_coords = sampling.index2points(coords, sampled_point_indices)
        return center_coords


