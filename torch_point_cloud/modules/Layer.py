import torch
from torch import nn

class Layers(nn.Module):
    def __init__(self, layer, norm, act):
        super(Layers, self).__init__()
        module_list = []
        module_list.append(layer)
        module_list.append(norm)
        if act is not None:
            module_list.append(act)

        self.net = nn.Sequential(
            *module_list
        )

    def forward(self, x):
        return self.net(x)

class Conv1DModule(Layers):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True), 
                 conv_args={}, bn_args={}):
        """
        Point-wise conv modules
        """
        conv = nn.Conv1d(in_channels, out_channels, 1, **conv_args)
        norm = nn.BatchNorm1d(out_channels, **bn_args)
        super().__init__(conv, norm, act)

class Conv2DModule(Layers):
    def __init__(self, in_channels, out_channels, act=nn.ReLU(inplace=True),
                 conv_args={}, bn_args={}):
        conv = nn.Conv2d(in_channels, out_channels, (1,1), **conv_args)
        norm = nn.BatchNorm2d(out_channels, **bn_args)
        super().__init__(conv, norm, act)

class LinearModule(Layers):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True),
                 linear_args={}, bn_args={}):
        layer = nn.Linear(in_features, out_features, **linear_args)
        norm = nn.BatchNorm1d(out_features, **bn_args)
        super().__init__(layer, norm, act)
