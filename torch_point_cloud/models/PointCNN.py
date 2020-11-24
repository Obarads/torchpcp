import torch
from torch import nn

from torch_point_cloud.modules.XConv import XConv

class PointCNN(nn.Module):
    def __init__(self, in_channel, out_channel, use_x_transform=True):
        super().__init__()

        



