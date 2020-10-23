import torch
from torch import nn
from torch.nn import functional as F

from torch_point_cloud.modules.Layer import MLP1D
from torch_point_cloud.modules.Sampling import (
    index_points, query_ball_point, sample_and_group, square_distance)
from torch_point_cloud.modules.PointNetSetAbstraction import(
    PointNetSetAbstraction, PointNetSetAbstractionMSG)
from torch_point_cloud.modules.PointNetFeaturePropagation import(
    PointNetFeaturePropagation)

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/2d08fa40635cc5eafd14d19d18e3dc646171910d/models/pointnet2_cls_ssg.py#L10
class PointNet2SSGClassification(nn.Module):
    def __init__(self, num_classes, in_channel=3):
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                          in_channel=in_channel, mlp=[]