import torch
from torch import nn
from torch.nn import functional as F

from torchpcp.modules.Layer import PointwiseConv1D, Linear

from torchpcp.modules.PointNetSetAbstraction import(
    PointNetSetAbstraction, PointNetSetAbstractionMSG)
from torchpcp.modules.PointNetFeaturePropagation2 import(
    PointNetFeaturePropagation)

from torchpcp.utils.monitor import timecheck

class PointNet2SSGClassification(nn.Module):
    """
    PointNet++ with SSG for Classification
    Parameters
    ----------
    num_classes:int
        number of classes
    point_feature_size:int
        size of input feature other than xyz
    """
    def __init__(self, num_classes, point_feature_size=0):
        in_channel = 3+point_feature_size
        self.sa1 = PointNetSetAbstraction(num_fps_points=512, radius=0.2, 
                                          num_bq_points=32, 
                                          init_in_channel=in_channel, 
                                          mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(num_fps_points=128, radius=0.4, 
                                          num_bq_points=64,
                                          init_in_channel=128 + 3,
                                          mlp=[128, 128, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(num_fps_points=None, radius=None, 
                                          num_bq_points=None,
                                          init_in_channel=256+3,
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
        self.sa1 = PointNetSetAbstraction(num_fps_points=1024, radius=0.1, 
                                          num_bq_points=32, 
                                          init_in_channel=in_channel, 
                                          mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(num_fps_points=256, radius=0.2, 
                                          num_bq_points=32,
                                          init_in_channel=64 + 3,
                                          mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(num_fps_points=64, radius=0.4, 
                                          num_bq_points=32,
                                          init_in_channel=128+3,
                                          mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(num_fps_points=16, radius=0.8, 
                                          num_bq_points=32,
                                          init_in_channel=256 + 3,
                                          mlp=[256, 256, 512], group_all=False)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128+in_channel, [128, 128, 128])

        self.output_layers = nn.Sequential(
            PointwiseConv1D(128, 128),
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


NPOINTS = [4096, 1024, 256, 64]
RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
NSAMPLE = [[16, 32], [16, 32], [16, 32], [16, 32]]
MLPS = [[[16, 16, 32], [32, 32, 64]], [[64, 64, 128], [64, 96, 128]],
        [[128, 196, 256], [128, 196, 256]], [[256, 256, 512], [256, 384, 512]]]
FP_MLPS = [[512, 512], [512, 512], [256, 256], [128, 128]]
CLS_FC = [128]
DP_RATIO = 0.5

# class PointNet2MSGSemanticSegmentation(nn.Module):
#     """
#     Parameters
#     ----------
#     point_feature_size:int
#         feature size other than xyz
#     """
#     def __init__(self, point_feature_size, num_points=NPOINTS, radius=RADIUS, 
#                  num_sample=NSAMPLE, mlps=MLPS, fp_mls=FP_MLPS, cls_fc=CLS_FC, 
#                  dp_ratio=DP_RATIO):
#         super().__init__()
        
#         self.sa_msg_modules = nn.ModuleList()
#         prev_feature_size = point_feature_size
#         sa_output_feature_size = []
#         for i in range(len(num_points)):
#             self.sa_msg_modules.append(
#                 PointNetSetAbstractionMSG(
#                     num_points[i],
#                     radius[i],
#                     num_sample[i],
#                     prev_feature_size,
#                     mlps[i]
#                 )
#             )
#             prev_feature_size = 0
#             for feature_size in mlps[i]: prev_feature_size += feature_size[-1]
#             sa_output_feature_size.append(prev_feature_size)

#         # for fp module
#         sa_output_feature_size = list(reversed(sa_output_feature_size))

#         self.fp_modules = nn.ModuleList()
#         prev_feature_size = sa_output_feature_size[0]
#         for i in range(len(fp_mls)):
#             input_feature_size = prev_feature_size
#             if len(sa_output_feature_size) > i+1:
#                 input_feature_size += sa_output_feature_size[i+1]
#             self.fp_modules.append(
#                 PointNetFeaturePropagation(
#                     input_feature_size,
#                     fp_mls[i]
#                 )
#             )
#             prev_feature_size = fp_mls[i][-1]

#         self.cls_fc = nn.ModuleList()
#         prev_feature_size = fp_mls[-1][-1]
#         for i in range(len(cls_fc)):
#             self.cls_fc.append(PointwiseConv1D(prev_feature_size, cls_fc[i]))
#             prev_feature_size = cls_fc[i]
#         # Not implemetation : dp_ratio

#     def forward(self, xyz, features):
#         sa_features_list = []
#         sa_xyz_list = []

#         sa_features_list.append(features)
#         sa_xyz_list.append(xyz)
#         for sa_msg in self.sa_msg_modules:
#             sa_xyz, sa_features = sa_msg(sa_xyz_list[-1], sa_features_list[-1])
#             sa_xyz_list.append(sa_xyz)
#             sa_features_list.append(sa_features)

#         # for fp module
#         sa_xyz_list = list(reversed(sa_xyz_list))
#         sa_features_list = list(reversed(sa_features_list))

#         prev_features = sa_features_list[0]
#         for i, fp in enumerate(self.fp_modules):
#             # don't concat intensity data
#             if len(sa_features_list)-1 > i+1:
#                 sa_features = sa_features_list[i+1]
#             else:
#                 sa_features = None
#             prev_features = fp(sa_xyz_list[i+1], sa_xyz_list[i], 
#                                sa_features, prev_features)

#         outpus = prev_features
#         for fc in self.cls_fc:
#             outpus = fc(outpus)

#         return outpus

class PointNet2MSGSemanticSegmentation(nn.Module):
    """
    Parameters
    ----------
    point_feature_size:int
        feature size other than xyz
    """
    def __init__(self, point_feature_size, num_points=NPOINTS, radius=RADIUS, 
                 num_sample=NSAMPLE, mlps=MLPS, fp_mls=FP_MLPS, cls_fc=CLS_FC, 
                 dp_ratio=DP_RATIO):
        super().__init__()
        
        # create sa modules
        self.sa_msg_modules = nn.ModuleList()
        prev_feature_size = point_feature_size
        sa_output_feature_size = []
        for i in range(len(num_points)):
            self.sa_msg_modules.append(
                PointNetSetAbstractionMSG(
                    num_points[i],
                    radius[i],
                    num_sample[i],
                    prev_feature_size,
                    mlps[i]
                )
            )
            prev_feature_size = 0
            for feature_size in mlps[i]: prev_feature_size += feature_size[-1]
            sa_output_feature_size.append(prev_feature_size)

        # create fp module
        self.fp_modules = nn.ModuleList()
        sa_output_feature_size = list(reversed(sa_output_feature_size))
        prev_feature_size = sa_output_feature_size[0]
        for i in range(len(fp_mls)):
            input_feature_size = prev_feature_size
            if len(sa_output_feature_size) > i+1:
                input_feature_size += sa_output_feature_size[i+1]
            self.fp_modules.append(
                PointNetFeaturePropagation(
                    input_feature_size,
                    fp_mls[i]
                )
            )
            prev_feature_size = fp_mls[i][-1] 

        # create branch
        self.cls_fc = nn.ModuleList()
        prev_feature_size = fp_mls[-1][-1]
        for i in range(len(cls_fc)):
            self.cls_fc.append(PointwiseConv1D(prev_feature_size, cls_fc[i]))
            prev_feature_size = cls_fc[i]
        # Not implemetation : dp_ratio

    def forward(self, xyz, features):
        sa_features_list = [features]
        sa_xyz_list = [xyz]

        t = timecheck()
        # sa
        for sa_msg in self.sa_msg_modules:
            sa_xyz, sa_features = sa_msg(sa_xyz_list[-1], sa_features_list[-1])
            sa_xyz_list.append(sa_xyz)
            sa_features_list.append(sa_features)
        t = timecheck(t, "sa")

        # fp
        prev_features = sa_features_list[-1]
        for i in range(len(self.fp_modules)):
            # don't concat intensity data
            curr_idx = -(i+1)
            next_idx = -(i+2)
            sa_features = sa_features_list[next_idx]
            prev_features = self.fp_modules[i](sa_xyz_list[next_idx], 
                                               sa_xyz_list[curr_idx], 
                                               sa_features, prev_features)
        t = timecheck(t, "fp")
        # branch
        outpus = prev_features
        for fc in self.cls_fc:
            outpus = fc(outpus)

        return outpus