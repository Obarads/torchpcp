import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule
from torch_point_cloud.modules.functional import sampling

from torch_point_cloud.modules.Layer import Conv2D

class XTransform(nn.Module):
    def __init__(self, in_channel, k):
        super().__init__()

        self.conv1 = Conv2D(in_channel, k*k, (1,k)) # [B, k*k, N, 1] # pf.conv2d is not this order
        self.conv2 = Conv2D(k*k, k*k, (1,1), conv_args={"groups":k}) # DepthwiseConv2D(k, k, (1, k)) & convert(x)
        self.conv3 = Conv2D(k*k, k*k, (1,1), act=None, conv_args={"groups":k}) # DepthwiseConv2D(k, k, (1, k), act=None) & convert(x)
        self.k = k

    def forward(self, x):
        """
        Parameter
        ---------
        x: [B, C, N, k]
            Inputs.

        Returns
        -------
        trans: [B, N, k, k]
            X-transformation matrix.
        """
        # B, C, N, k = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        trans = self.to_trans(x)

        return trans

    def to_trans(self, x):
        B, kk, N, _ = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(B, N, self.k, self.k).contiguous()
        return x
