import numpy as np

import torch
from torch import nn

from torch_point_cloud.modules.Layer import Layers

class Conv1DModule(Layers):
    def __init__(self, in_channels, out_channels, 
                 act=nn.ReLU(inplace=True)):
        conv = nn.Conv1d(in_channels, out_channels, 1)
        nn.init.xavier_uniform_(conv.weight)
        norm = nn.BatchNorm1d(out_channels)
        super().__init__(conv, norm, act)

class LinearModule(Layers):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
        layer = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(layer.weight)
        norm = nn.BatchNorm1d(out_features)
        super(Linear, self).__init__(layer, norm, act)

class InputTransformNet(nn.Module):
    """
    Transform network for XYZ coordinate.
    """
    def __init__(self):
        super(InputTransformNet, self).__init__()

        # layers before a max-pooling
        self.encoder = nn.Sequential(
            Conv1DModule(3, 64),
            Conv1DModule(64, 128),
            Conv1DModule(128, 1024)
        )

        # last layer setting
        fc = nn.Linear(256, 3*3, bias=True)
        # nn.init.zeros_(fc.weight)
        fc.bias = nn.Parameter(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], 
                               dtype=torch.float32))

        # layers after a max-pooling
        self.decoder = nn.Sequential(
            LinearModule(1024, 512),
            LinearModule(512, 256),
            fc
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.decoder(x)
        x = x.view(-1, 3, 3)
        return x

class FeatureTransformNet(nn.Module):
    """
    Transform network for features.
    """
    def __init__(self, k=64):
        super(FeatureTransformNet, self).__init__()

        # layers before a max-pooling
        self.encoder = nn.Sequential(
            Conv1DModule(k, 64),
            Conv1DModule(64, 128),
            Conv1DModule(128, 1024)
        )

        # last layer setting
        fc = nn.Linear(256, k*k, bias=True)
        # nn.init.zeros_(fc.weight)
        fc.bias = nn.Parameter(torch.tensor(np.eye(k).flatten(), 
                               dtype=torch.float32))

        # layers after a max-pooling
        self.decoder = nn.Sequential(
            LinearModule(1024, 512),
            LinearModule(512, 256),
            fc
        )

        # args
        self.k = k

    def forward(self, x):
        x = self.encoder(x)
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.decoder(x)
        x = x.view(-1, self.k, self.k)
        return x
