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
        in_channel = 3+point_feature_size
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                          in_channel=in_channel, 
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

class PointNet2SSGSemanticSegmentation(nn.Module):
    """
    PointNet++ with SSG for Semantic segmentation
    Parameters
    ----------
    num_classes:int
        number of classes
    point_feature_size:int
        feature size other than xyz
    """
    def __init__(self, num_classes, point_feature_size=0):
        super().__init__()
        in_channel = 3+point_feature_size
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, 
                                          in_channel=in_channel, 
                                          mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32,
                                          in_channel=64 + 3,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32,
                                          in_channel=128+3,
                                          mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32,
                                          in_channel=256 + 3,
                                          mlp=[256, 256, 512], group_all=False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128+point_feature_size, [128, 128, 128])

        self.output_layers = nn.Sequential(
            MLP1D(128, 128),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )

        self.point_feature_size = point_feature_size
    
    def forward(self, inputs):
        if self.point_feature_size > 0:
            l0_xyz = inputs[:, :3, :]
            l0_points = inputs[:, 3:, :]
        else:
            l0_xyz = inputs
            l0_points = None

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.output_layers(l0_points)

        return x


