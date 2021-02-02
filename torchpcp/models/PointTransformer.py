import torch
from torch import nn

from torchpcp.modules.PointTransformerBlock import (
    PointTransformerBlock, TransitionDown, NonTrasition)
from torchpcp.modules.Layer import PointwiseConv1D, Linear

class TDPT(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, pt_mid_channel_size,
                 coord_channel_size, td_k, num_sampling, pt_k):
        super().__init__()

        self.td = TransitionDown(in_channel_size, out_channel_size, td_k, num_sampling)
        self.pt = PointTransformerBlock(out_channel_size, pt_mid_channel_size, 
                                        coord_channel_size, pt_k)

    def forward(self, x, p1):
        x, p2 = self.td(x, p1)
        y = self.pt(x, p2)
        return y, p2

class NTPT(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, pt_mid_channel_size,
                 coord_channel_size, nt_k, pt_k):
        super().__init__()

        self.nt = NonTrasition(in_channel_size, out_channel_size, nt_k)
        self.pt = PointTransformerBlock(out_channel_size, pt_mid_channel_size,
                                        coord_channel_size, pt_k)
    
    def forward(self, x, p1):
        x, p2 = self.nt(x, p1) # p2 = p1
        y = self.pt(x, p2)
        return y, p2

class MLPPT(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, pt_mid_channel_size,
                 coord_channel_size, pt_k):
        super().__init__()

        self.pwc = PointwiseConv1D(in_channel_size, out_channel_size, conv_args={"bias": False})
        self.pt = PointTransformerBlock(out_channel_size, pt_mid_channel_size,
                                        coord_channel_size, pt_k)
    
    def forward(self, x, p1):
        x = self.pwc(x)
        y = self.pt(x, p1)
        return y, p1

class PointTransformerClassification(nn.Module):
    def __init__(
        self, 
        in_channel_size,
        in_num_point,
        coord_channel_size,
        num_points,
        encoder_channel_sizes,
        bottleneck_ratio,
        num_k_neighbors,
        decoder_channel_sizes

    ):
        super().__init__()

        # Create encoder layers.
        encoder = []
        prev_channel_size = in_channel_size
        prev_num_point = in_num_point
        for i in range(len(encoder_channel_sizes)):
            out_channel_size = encoder_channel_sizes[i]
            num_point = num_points[i]
            k = num_k_neighbors[i]
            if prev_num_point == num_point:
                encoder.append(MLPPT(prev_channel_size, out_channel_size, 
                                     out_channel_size//bottleneck_ratio,
                                     coord_channel_size, k))
            elif prev_num_point > num_point:
                encoder.append(TDPT(prev_channel_size, out_channel_size, 
                                    out_channel_size//bottleneck_ratio,
                                    coord_channel_size, k, num_point, k))
            else:
                raise ValueError("Number of points in the encoder (encoder_channel_sizes) must not be more than the previous layer.")
            prev_channel_size = out_channel_size
            prev_num_point = num_point
        self.encoder = nn.ModuleList(encoder)

        # Create decoder layers.
        decoder = []
        for i in range(len(decoder_channel_sizes)):
            out_channel_size = decoder_channel_sizes[i]
            if i == len(decoder_channel_sizes)-1:
                decoder.append(nn.Linear(prev_channel_size, out_channel_size, bias=False))
            elif i == "d":
                decoder.append(nn.Dropout())
            else:
                decoder.append(Linear(prev_channel_size, out_channel_size, linear_args={"bias": False}))
            prev_channel_size = out_channel_size
        # self.decoder = nn.ModuleList(decoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, coords):
        for enc in self.encoder:
            x, coords = enc(x, coords)

        x = torch.mean(x, dim=-1)

        y = self.decoder(x)

        return y

