import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule
from torch_point_cloud.modules.functional.sampling import (
    index2points,
    k_nearest_neighbors
)

class EdgeConv(nn.Module):
    def __init__(self, in_channel, out_channel, k, memory_saving=False):
        super().__init__()
        self.conv = Conv2DModule(
            in_channel, 
            out_channel, 
            act=nn.LeakyReLU(negative_slope=0.2),
            conv_args={"bias":False}
        )
        self.k = k
        self.memory_saving = memory_saving

    def forward(self, x):
        x = get_graph_feature(x, self.k, memory_saving=self.memory_saving)
        x = self.conv(x)
        x = torch.max(x, dim=-1, keepdim=False)[0]
        return x

def get_graph_feature(x, k=20, memory_saving=False):
    B, C, N = x.shape
    k_idx, _ = k_nearest_neighbors(x, x, k, memory_saving=memory_saving)
    feature = index2points(x, k_idx)
    x = x.view(B, C, N, 1).repeat(1, 1, 1, k)
    # x = torch.unsqueeze(x, dim=-1)

    x = torch.cat((feature-x, x), dim=1)

    return x
