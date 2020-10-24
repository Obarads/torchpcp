import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from torch_point_cloud.modules.Layer import MLP1D, Linear
from torch_point_cloud.modules.PointNetSetAbstraction import PointNetSetAbstraction
from torch_point_cloud.modules.PointNetFeaturePropagation import PointNetFeaturePropagation
from torch_point_cloud.modules.ASIS import ASIS

class PointNet2ASIS(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2ASIS, self).__init__()

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6+3, [32, 32, 64], None)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64+3, [64, 64, 128], None)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128+3, [128, 128, 256], None)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256+3, [256, 256, 512], None)

        self.fp_sem4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp_sem3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp_sem2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp_sem1 = PointNetFeaturePropagation(128+6, [128, 128, 128])

        self.fp_ins4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp_ins3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp_ins2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp_ins1 = PointNetFeaturePropagation(128+6, [128, 128, 128])

        self.sem_fc = MLP1D(128, 128) # for F_SEM
        self.ins_fc = MLP1D(128, 128) # for F_INS

        self.asis = ASIS(128, num_classes, 128, 5, 30)

    def forward(self, x):
        l0_points = x[:, 3:, :]
        l0_xyz = x[:, :3, :]
 
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points_sem = self.fp_sem4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_sem = self.fp_sem3(l2_xyz, l3_xyz, l2_points, l3_points_sem)
        l1_points_sem = self.fp_sem2(l1_xyz, l2_xyz, l1_points, l2_points_sem)
        l0_points_sem = self.fp_sem1(l0_xyz, l1_xyz, l0_points, l1_points_sem)

        l3_points_ins = self.fp_ins4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_ins = self.fp_ins3(l2_xyz, l3_xyz, l2_points, l3_points_ins)
        l1_points_ins = self.fp_ins2(l1_xyz, l2_xyz, l1_points, l2_points_ins)
        l0_points_ins = self.fp_ins1(l0_xyz, l1_xyz, l0_points, l1_points_ins)

        f_sem = self.sem_fc(l0_points_sem)
        f_ins = self.ins_fc(l0_points_ins)

        p_sem, e_ins = self.asis(f_sem, f_ins)

        return p_sem, e_ins
