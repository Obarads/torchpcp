import torch
from torch import nn
from torch.nn import functional as F

from torchpcp.modules.EdgeConv import EdgeConv
from torchpcp.modules.Layer import LinearModule, Conv1DModule

class DGCNNClassification(nn.Module):
    def __init__(self, out_channel, k, emb_dims, dropout_p):
        super().__init__()

        self.conv1 = EdgeConv(6, 64, k)
        self.conv2 = EdgeConv(64*2, 64, k)
        self.conv3 = EdgeConv(64*2, 128, k)
        self.conv4 = EdgeConv(128*2, 256, k)
        self.conv5 = Conv1DModule(512, emb_dims)

        self.linear1 = LinearModule(emb_dims*2, 512, 
                              act=nn.LeakyReLU(negative_slope=0.2), 
                              linear_args={"bias": False})
        self.dp1 = nn.Dropout(p=dropout_p)
        
        self.linear2 = LinearModule(512, 256,
                              act=nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=dropout_p)

        self.linear3 = nn.Linear(256, out_channel)

    def forward(self, x):
        B, C, N = x.shape

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.conv5(x)

        x_amp = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x_aap = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x_amp, x_aap), 1)

        x = self.linear1(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

