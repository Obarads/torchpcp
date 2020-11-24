import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule, DepthwiseSeparableConv2D, LinearModule
from torch_point_cloud.modules.functional import sampling
from torch_point_cloud.modules.XTransformation import XTransform

class XConv(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        k, 
        C,
        depth_multiplier,
        qrs_in_channel = 0,
        fts_in_channel=0,
        use_x_transformation=True, 
        memory_saving=False
    ):
        super().__init__()

        c_pts_fts = None


        self.mlp_d = nn.Sequential(
            Conv2DModule(in_channel, c_pts_fts), # pf.dense is not this order
            Conv2DModule(c_pts_fts, c_pts_fts)
        )

        if use_x_transformation:
            self.x_trans = XTransform(c_pts_fts+fts_in_channel, k)

        self.conv1 = DepthwiseSeparableConv2D(c_pts_fts+fts_in_channel, C, depth_multiplier, (1, k))

        if qrs_in_channel > 0: # if with_global (https://github.com/yangyanli/PointCNN/blob/91fde862b1818aec305dacafe7438d8f1ca1d1ea/pointcnn.py#L47)
            self.linear1 = nn.Sequential(
                LinearModule(qrs_in_channel, C//4),
                LinearModule(C//4, C//4)
            )

        self.k = k
        self.qrs_in_channel = qrs_in_channel
        self.fts_in_channel = fts_in_channel
        self.use_x_transformation = use_x_transformation
        self.memory_saving = memory_saving


    def forward(self, fts, center_xyz, center_points, xyz, points):
        knn_indexes, _ = sampling.k_nearest_neighbors(center_xyz, xyz, self.k, self.memory_saving)
        knn_points = sampling.index2points(points, knn_indexes)
        knn_local_features = sampling.localize(center_points, knn_points)
        feature_d = self.mlp_d(knn_local_features)

        if self.fts_in_channel == 0:
            feature_a = feature_d
        else:
            prev_ = sampling.index2points(fts, knn_indexes)
            feature_a = torch.cat([feature_d, prev_], dim=1) # [B, C+add_C, N, k]

        if self.use_x_transformation:
            trans = self.x_trans(feature_a)
            trans = trans.permute(0, 2, 3, 1)
            feature_a = feature_a.permute(0, 2, 3, 1)
            fx = torch.matmul(trans, feature_a)
            fx = fx.permute(0, 3, 1, 2)
        else:
            fx = feature_a

        if self.qrs_in_channel > 0:
            fts_global = self.linear1(center_xyz)
            res = torch.can([fts_global, fx], dim=1)
        else:
            res = fx
        
        return res



