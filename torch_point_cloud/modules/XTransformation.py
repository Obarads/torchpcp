import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule
from torch_point_cloud.modules.functional import sampling

from torch_point_cloud.modules.Layer import PointwiseConv2D, DepthwiseConv2D, Layers

# class DepthwiseConv2D(Layers):
#     def __init__(self, in_channel, out_channel, k, act=nn.ReLU(inplace=True)):
#         conv = nn.Conv2d(in_channel, out_channel, (1, k))
#         norm = nn.BatchNorm2d(out_channel)
        # super().__init__(conv, norm, act)

class Conv2D(Layers):
    def __init__(self, in_channel, out_channel, kernel_size, act=nn.ReLU(inplace=True)):
        conv = nn.Conv2d(in_channel, out_channel, kernel_size)
        norm = nn.BatchNorm2d(out_channel)
        super().__init__(conv, norm, act)

class XTransform(nn.Module):
    def __init__(self, in_channel, k):
        self.conv1 = Conv2D(in_channel, k*k, (1,k)) # [B, k*k, N, 1] # pf.conv2d is not this order
        self.conv2 = DepthwiseConv2D(k, k, (1, k))
        self.conv3 = DepthwiseConv2D(k, k, (1, k), act=None)

        self.k = k

    def forward(self, x):
        """
        x: [B, C, N, k]
        """
        B, C, N, k = x.shape
        x = self.conv1(x)
        x = self.convert(x)
        x = self.conv2(x)
        x = self.convert(x)
        x = self.conv3(x)
        trans = self.convert(x)

        return trans

    # bad impl.
    def convert(self, x):
        """
        Parameters
        ----------
        x: [B, k*k, N, 1]

        Outputs
        -------
        trans: [B, k, N, k]
        """
        B, _, N, _ = x.shape
        x = x.permute(0,2,3,1)
        x = x.view(B, N, self.k, self.k) 
        x = x.permute(0, 3, 1, 2)
        return x

