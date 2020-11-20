import torch
from torch import nn

from torch_point_cloud.modules.functional import sampling

class XConv(nn.Module):
    def __init__(self, in_channel, out_channel, k, memory_saving=False):
        super().__init__()

        self.k = k
        self.memory_saving = memory_saving

    def forward(self, center_xyz, xyz, points):
        sampling.k_nearest_neighbors(center_xyz, xyz, k, self.memory_saving)
        return


