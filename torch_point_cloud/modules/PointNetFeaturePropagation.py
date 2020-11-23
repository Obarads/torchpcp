import torch
from torch import nn
from torch.nn import functional as F

from torch_point_cloud.modules.Layer import Conv1DModule
from torch_point_cloud.modules.functional.sampling import (
    k_nearest_neighbors,
    index2points
)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, init_in_channel, mlp):
        super().__init__()

        layers = []
        in_channel = init_in_channel
        for out_channel in mlp:
            layers.append(Conv1DModule(in_channel, out_channel))
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1, xyz2, points1, points2):
        B, C, N = xyz1.shape
        _, _, S = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            idxs, dists = k_nearest_neighbors(xyz1, xyz2, 3, memory_saving=True)

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            a = index2points(points2, idxs)
            w = weight.view(B, 1, N, 3)
            aw = a * w
            interpolated_points = torch.sum(aw, dim=3)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        new_points = self.mlp(new_points)

        return new_points


