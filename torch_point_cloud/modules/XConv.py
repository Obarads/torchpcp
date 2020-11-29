import torch
from torch import nn

from torch_point_cloud.modules.Layer import PointwiseConv2D, DepthwiseSeparableConv2D, LinearModule
from torch_point_cloud.modules.functional import sampling
from torch_point_cloud.modules.XTransformation import XTransform

# class Linear(nn.Module):
#     def __init__(self, in_channel, out_channel, act=nn.ReLU()):
#         super().__init__()
#         self.layer = nn.Linear(in_channel, out_channel)
#         self.bn = nn.BatchNorm2d(out_channel)
#         self.act = act

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1).contiguous()
#         x = self.layer(x)
#         x = x.permute(0, 3, 1, 2).contiguous()
#         x = self.bn(x)
#         if self.act is not None:
#             x = self.act(x)
#         return x

class XConv(nn.Module):
    def __init__(
        self, 
        coord2feature_channel, # C_pts_fts
        point_feature_size, # features other than xyz
        out_channel, # C
        k,# K
        dilation, # dilation rate, D
        depth_multiplier,
        with_global=False,
        use_x_transformation=True, 
        memory_saving=False
    ):
        super().__init__()

        # self.mlp_d = nn.Sequential(
        #     Linear(3, coord2feature_channel),
        #     Linear(coord2feature_channel, coord2feature_channel)
        # )
        self.mlp_d = nn.Sequential(
            PointwiseConv2D(3, coord2feature_channel),
            PointwiseConv2D(coord2feature_channel, coord2feature_channel)
        )

        if use_x_transformation:
            self.x_trans = XTransform(
                coord2feature_channel + point_feature_size,
                k
            )

        self.conv1 = DepthwiseSeparableConv2D(
            coord2feature_channel + point_feature_size, 
            out_channel, 
            depth_multiplier, 
            (1, k)
        )

        if with_global: # (https://github.com/yangyanli/PointCNN/blob/91fde862b1818aec305dacafe7438d8f1ca1d1ea/pointcnn.py#L47)
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
        knn_local_coords = sampling.localize(center_coords, knn_coords) # (B,C,N,k)
        
        feature_d = self.mlp_d(knn_local_coords)

        if self.point_feature_size == 0:
            feature_a = feature_d
        else:
            knn_features = sampling.index2points(features, knn_indexes)
            feature_a = torch.cat([feature_d, knn_features], dim=1) # [B, C+add_C, N, k]

        if self.use_x_transformation:
            trans = self.x_trans(feature_a)
            fx = self.transform(feature_a, trans)
        else:
            fx = feature_a

        fx = self.conv1(fx)
        fx = torch.squeeze(fx, dim=3).contiguous()

        if self.with_global:
            fts_global = self.linear1(center_coords)
            res = torch.can([fts_global, fx], dim=1)
        else:
            res = fx
        
        return res

    def transform(self, x, trans):
        """
        Parameters
        ----------
        x: [B, C, N, k]
            features
        trans: [B, N, k, k]
            trans

        Returns
        -------
        transformed_x: [B, C, N, k]
            transformed_x
        """
        x = x.permute(0, 2, 3, 1).contiguous()
        transformed_x = torch.matmul(trans, x).contiguous()
        transformed_x = transformed_x.permute(0, 3, 1, 2).contiguous()
        return transformed_x

