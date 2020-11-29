import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule
from torch_point_cloud.modules.functional import sampling

from torch_point_cloud.modules.Layer import Conv2D, PointwiseConv2D
from torch_point_cloud.modules.Layer import DepthwiseConv2D

class XTransform(nn.Module):
    def __init__(self, in_channel, k):
        super().__init__()

        self.conv1 = Conv2D(in_channel, k*k, (1,k)) # [B, k*k, N, 1] # pf.conv2d is not this order
        # self.conv2 = DepthwiseConv2D(k, k, (1, k))
        # self.conv3 = DepthwiseConv2D(k, k, (1, k), act=None)
        self.conv2 = Conv2D(k*k, k*k, (1,1), conv_args={"groups":k})
        self.conv3 = Conv2D(k*k, k*k, (1,1), conv_args={"groups":k})
        self.k = k

    def forward(self, x):
        """
        x: [B, C, N, k]
        """
        # B, C, N, k = x.shape
        x = self.conv1(x)
        # x = self.convert(x)
        # x = self.conv2(x)
        # x = self.convert(x)
        # x = self.conv3(x)
        x = self.conv2(x)
        x = self.conv3(x)
        trans = self.to_trans(x)

        return trans

    def to_trans(self, x):
        B, kk, N, _ = x.shape
        x = x.permute(0,2,3,1).contiguous()
        x = x.view(B, N, self.k, self.k).contiguous()
        return x

    # bad impl.
    # output warning 
    # def convert(self, x):
    #     """
    #     Parameters
    #     ----------
    #     x: [B, k*k, N, 1]

    #     Outputs
    #     -------
    #     trans: [B, k, N, k]
    #     """
    #     B, _, N, _ = x.shape
    #     x = x.permute(0,2,3,1).contiguous()
    #     x = x.view(B, N, self.k, self.k).contiguous()
    #     x = x.permute(0, 3, 1, 2).contiguous()
    #     return x

