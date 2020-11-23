import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule
from torch_point_cloud.modules.functional import sampling

from torch_point_cloud.modules.Layer import Layers

class DepthwiseConv2D(Layers):
    def __init__(self, in_channel, out_channel, k, act=nn.ReLU(inplace=True)):
        conv = nn.Conv2d(in_channel, out_channel, (1, k))
        norm = nn.BatchNorm2d(out_channel)
        super().__init__(conv, norm, act)

class Conv2D(Layers):
    def __init__(self, in_channel, out_channel, kernel_size, act=nn.ReLU(inplace=True)):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size)
        norm = nn.BatchNorm2d(out_channel)
        super().__init__(conv, norm, act)

class XTransform(nn.Module):
    def __init__(self, in_channel, k):
        # self.conv1 = Conv2DModule(in_channel, k) # pf.conv2d is not this order
        # self.conv2 = Conv2DModule(k, k)
        # self.conv3 = Conv2DModule(k, k)
        # self.conv4 = Conv2DModule(k, k, act=None)

        self.conv1 = Conv2D(in_channel, k*k, (1,k)) # [B, k*k, N, 1] # pf.conv2d is not this order
        self.conv2 = DepthwiseConv2D
            # DepthwiseConv2D(),
            # Conv2DModule(k, k),
            # Conv2DModule(k, k, act=None),

    def forward(self, x):
        """
        x: [B, C, N, k]
        """
        x = self.conv1(x)
        x_kk = x.permute()
        return trans



