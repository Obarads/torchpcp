import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule, DepthwiseSeparableConv2D, LinearModule
from torch_point_cloud.modules.functional import sampling
from torch_point_cloud.modules.XTransformation import XTransform

class XConv(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel, # C
        k,# K
        dilation, # dilation rate, D
        depth_multiplier,
        with_global = False,
        point_feature_size=0, # features other than xyz
        use_x_transformation=True, 
        memory_saving=False
    ):
        super().__init__()

        c_pts_fts = None

        self.mlp_d = nn.Sequential(
            LinearModule(in_channel, c_pts_fts),
            LinearModule(c_pts_fts, c_pts_fts)
        )

        # self.mlp_d = nn.Sequential(
        #     Conv2DModule(in_channel, c_pts_fts), # pf.dense is not this order
        #     Conv2DModule(c_pts_fts, c_pts_fts)
        # )

        if use_x_transformation:
            self.x_trans = XTransform(c_pts_fts+point_feature_size, k)

        self.conv1 = DepthwiseSeparableConv2D(c_pts_fts+point_feature_size, out_channel, depth_multiplier, (1, k))

        if with_global > 0: # if with_global (https://github.com/yangyanli/PointCNN/blob/91fde862b1818aec305dacafe7438d8f1ca1d1ea/pointcnn.py#L47)
            self.linear1 = nn.Sequential(
                LinearModule(3, out_channel//4),
                LinearModule(out_channel//4, out_channel//4)
            )

        self.k = k
        self.dilation = dilation
        self.with_global = with_global
        self.point_feature_size = point_feature_size
        self.use_x_transformation = use_x_transformation
        self.memory_saving = memory_saving


    def forward(self, center_coords, coords, features):
        knn_indexes, _ = sampling.k_nearest_neighbors(center_coords, coords, self.k * self.dilation, self.memory_saving)
        knn_indexes = knn_indexes[:, :, ::self.dilation]
        knn_coords = sampling.index2points(coords, knn_indexes)
        knn_local_coords = sampling.localize(center_coords, knn_coords)
        feature_d = self.mlp_d(knn_local_coords)

        if self.point_feature_size == 0:
            feature_a = feature_d
        else:
            knn_features = sampling.index2points(features, knn_indexes)
            feature_a = torch.cat([feature_d, knn_features], dim=1) # [B, C+add_C, N, k]

        if self.use_x_transformation:
            trans = self.x_trans(feature_a)
            trans = trans.permute(0, 2, 3, 1)
            feature_a = feature_a.permute(0, 2, 3, 1)
            fx = torch.matmul(trans, feature_a)
            fx = fx.permute(0, 3, 1, 2)
        else:
            fx = feature_a

        if self.qrs_in_channel > 0:
            fts_global = self.linear1(center_coords)
            res = torch.can([fts_global, fx], dim=1)
        else:
            res = fx
        
        return res



