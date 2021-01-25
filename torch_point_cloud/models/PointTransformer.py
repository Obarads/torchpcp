import torch
from torch import nn

from torch_point_cloud.modules.PointTransformerBlock import (
    PointTransformerBlock,
    TransitionDown
)

from torch_point_cloud.modules.Layer import PointwiseConv1D, Linear

class TDPT(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, coord_channel_size, td_k, num_sampling, pt_k):
        super().__init__()

        self.td = TransitionDown(in_channel_size, out_channel_size, td_k, num_sampling)
        self.pt = PointTransformerBlock(out_channel_size, out_channel_size//4, 
                                        coord_channel_size, pt_k)

    def forward(self, x, p1):
        x, p2 = self.td(x, p1)
        y = self.pt(x, p2)
        return y, p2

IN_CHANNEL_SIZE = 6
IN_OUT_CHANNEL_SIZE = 32
IN_PT_K = 16

COORD_CHANNEL_SIZE=3
DATASET_NUM_POINTS = 1024

NUM_POINTS = [DATASET_NUM_POINTS//4, DATASET_NUM_POINTS//16, DATASET_NUM_POINTS//64, 
              DATASET_NUM_POINTS//256]
OUT_CHANNEL_SIZES = [64, 128, 256, 512]
TD_KS = [16, 16, 16, 16] # KNN for transition down
PT_KS = [16, 16, 16, 16] # KNN for point transformer

OUT_OUT_CHANNEL_SIZES = [512, 256, 40]

class PointTransformerClassification(nn.Module):
    def __init__(
        self, 
        in_channel_size=IN_CHANNEL_SIZE, 
        in_out_channel_size=IN_OUT_CHANNEL_SIZE,
        in_pt_k = IN_PT_K,
        num_points=NUM_POINTS,
        out_channel_sizes=OUT_CHANNEL_SIZES,
        td_ks=TD_KS,
        pt_ks=PT_KS,
        out_out_channel_sizes=OUT_OUT_CHANNEL_SIZES,
        coord_channel_size=COORD_CHANNEL_SIZE
    ):
        super().__init__()

        self.input_mlp = PointwiseConv1D(in_channel_size, in_out_channel_size)
        self.module_1 = PointTransformerBlock(in_out_channel_size, in_out_channel_size//4,
                                              coord_channel_size, in_pt_k)

        in_channel_size = in_out_channel_size
        self.encoder = nn.ModuleList()
        for i in range(len(num_points)):
            num_point = num_points[i]
            out_channel_size = out_channel_sizes[i]
            td_k = td_ks[i]
            pt_k = pt_ks[i]
            self.encoder.append(TDPT(in_channel_size, out_channel_size, 
                                     coord_channel_size, td_k, num_point, pt_k))
            in_channel_size = out_channel_size

        decoder = []
        for i in range(len(out_out_channel_sizes)):
            out_out_channel_size = out_out_channel_sizes[i]
            if i == len(out_out_channel_sizes)-1:
                layers = nn.Linear(in_channel_size, out_out_channel_size)
            else:
                layers = Linear(in_channel_size, out_out_channel_size)
            decoder.append(layers)
            in_channel_size = out_out_channel_size
        
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, coords):
        x = self.input_mlp(x)
        x = self.module_1(x, coords)

        for enc in self.encoder:
            x, coords = enc(x, coords)

        # print(x.shape)
        x = torch.mean(x, dim=-1)

        y = self.decoder(x)

        return y

