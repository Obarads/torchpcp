import torch
from torch import nn
from torch.nn import functional as F

from torch_point_cloud.modules.Layer import MLP1D, Linear
from torch_point_cloud.modules.Sampling import (
    index_points, query_ball_point, sample_and_group, square_distance)
from torch_point_cloud.modules.PointNetSetAbstraction import(
    PointNetSetAbstraction, PointNetSetAbstractionMSG)
from torch_point_cloud.modules.PointNetFeaturePropagation import(
    PointNetFeaturePropagation)

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/2d08fa40635cc5eafd14d19d18e3dc646171910d/models/pointnet2_cls_ssg.py#L10
class PointNet2SSGClassification(nn.Module):
    """
    PointNet++ with SSG for Classification
    Parameters
    ----------
    num_classes:int
        number of classes
    point_feature_size:int
        feature size other than xyz
    """
    def __init__(self, num_classes, point_feature_size=0):
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                          in_channel=point_feature_size, 
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                          in_channel=128 + 3,
                                          mlp=[128, 128, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=256+3,
                                          mlp=[256,512, 1024], group_all=True)
        
        self.decoder = nn.Sequential(
            Linear(1024, 512),
            nn.Dropout(0.5),
            Linear(512, 256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes
        self.point_feature_size = point_feature_size

    def forward(self, inputs):
        B, C, N = inputs.shape
        
        if self.point_feature_size > 0:
            xyz = inputs[:, :3, :]
            features = inputs[:, 3:, :]
        else:
            xyz = inputs
            features = None

        l1_xyz, l1_points = self.sa1(xyz, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.decoder(x)

        return x




