import torch
from torch import nn

from torch_point_cloud.modules.Layer import Conv2DModule
from torch_point_cloud.modules.functional import sampling
from torch_point_cloud.modules.XTransformation import XTransform

class XConv(nn.Module):
    def __init__(
        self, 
        in_channel, 
        out_channel, 
        k, 
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

        self.k = k
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
            feature_a = torch.cat([feature_d, prev_], dim=1)

        if self.use_x_transformation:
            trans = self.x_trans(feature_a)
            fx = torch.matmul(trans, feature_a)
        else:
            fx = feature_a

        fts_


        
        return


