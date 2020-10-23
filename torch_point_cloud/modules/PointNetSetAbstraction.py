import torch
from torch import nn
from torch.nn import functional as F

from torch_point_cloud.models.modules.layer import MLP2D
from torch_point_cloud.models.modules.sampling import (
    index_points, farthest_point_sample, query_ball_point, sample_and_group, 
    sample_and_group_all)

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L166
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMSG(nn.Module):
    def __init__(self, num_points, radius_list, num_sample_list, 
                 init_in_channels, mlp_list):
        super(PointNetSetAbstractionMSG, self).__init__()
        
        self.mlp_list = nn.ModuleList()
        for mlp in mlp_list:
            layers = []
            in_channels = init_in_channels + 3
            for out_channels in mlp:
                layers.append(MLP2D(in_channels, out_channels))
                in_channels = out_channels
            self.mlp_list.append(nn.Sequential(*layers))

        self.num_points = num_points
        self.radius_list = radius_list
        self.num_sample_list = num_sample_list

    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L210
    def forward(self, xyz, points):
        # preprocessing
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.num_points
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            # Sampling layer
            K = self.num_sample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)

            # Grouping layer
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat(
                    [grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]

            # PointNet layer
            grouped_points = self.mlp_list[i](grouped_points)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        # MSG Concat
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

