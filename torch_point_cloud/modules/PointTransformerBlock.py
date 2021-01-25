import torch
from torch import nn

from torch_point_cloud.modules.functional.nns import k_nearest_neighbors
from torch_point_cloud.modules.functional.other import index2points, localize
from torch_point_cloud.modules.functional.sampling import furthest_point_sampling
# from torch_point_cloud.modules.Layer import PointwiseConv2D

from einops import repeat

class PointTransformerLayer(nn.Module):
    """
    """
    def __init__(self, in_channel_size, out_channel_size, coords_channel_size, k):
        super().__init__()
        # phi (pointwise feature transformations)
        self.linear_phi = nn.Conv1d(in_channel_size, out_channel_size, 1)
        # psi (pointwise feature transformations)
        self.linear_psi = nn.Conv1d(in_channel_size, out_channel_size, 1)
        # alpha (pointwise feature transformation)
        self.linear_alpha = nn.Conv1d(in_channel_size, out_channel_size, 1)

        #  gamma (mapping function)
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(out_channel_size, out_channel_size, (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel_size, out_channel_size, (1,1))
        )

        # rho (normalization function)
        self.normalization_rho = nn.Softmax(dim=-1)

        # delta (potition encoding)
        self.pe_delta = PositionEncoding(coords_channel_size, out_channel_size)
        self.k = k

        
    def forward(self, features, coords):
        """
        Parameters
        ----------
        features: torch.tensor (B, C, N)
        """

        outputs_phi = self.linear_phi(features)
        outputs_psi = self.linear_psi(features)
        outputs_alpha = self.linear_alpha(features)

        # Get space between features of points.
        knn_indices, _ = k_nearest_neighbors(features, features, self.k)
        knn_outputs_psi = index2points(outputs_psi, knn_indices)
        features_space = localize(outputs_phi, knn_outputs_psi) * -1

        # Get delta.
        outputs_delta = self.pe_delta(coords, knn_indices)

        # Get gamma outputs.
        outputs_gamma = self.mlp_gamma(features_space + outputs_delta)

        # Get rho outputs.
        outputs_rho = self.normalization_rho(outputs_gamma)

        # \alpha(x_j) + \delta
        # print(outputs_alpha.shape, outputs_delta.shape)
        outputs_alpha = repeat(outputs_alpha, 'b c n -> b c n k', k=self.k) # really?????
        outputs_alpha_delta = outputs_alpha + outputs_delta

        # compute value with hadamard product
        # outputs_hp = outputs_rho * outputs_alpha_delta
        # aggregation outputs
        # outputs_aggregation = torch.sum(outputs_hp, dim=-1)
        outputs_aggregation = torch.einsum('b c n k, b c n k -> b c n', outputs_rho, outputs_alpha_delta)
        # print(outputs_aggregation == torch.einsum('b c n k, b c n k -> b c n', outputs_rho, outputs_alpha_delta))


        return outputs_aggregation

class PositionEncoding(nn.Module):
    def __init__(self, in_channel_size, out_channel_size):
        super().__init__()
        # theta (encoding function)
        self.mlp_theta = nn.Sequential(
            nn.Conv2d(in_channel_size, out_channel_size, (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel_size, out_channel_size, (1,1))
        )
        # self.k = k

    def forward(self, coords, knn_indices):
        """
        Parameters
        ----------
        coords: torch.tensor (B, C, N)
        """

        # Get spaces between points.
        # knn_indices = k_nearest_neighbors(coords, coords, self.k)
        knn_coords = index2points(coords, knn_indices)
        coords_space = localize(coords, knn_coords) * -1

        # Use theta.
        outputs = self.mlp_theta(coords_space)

        return outputs

class PointTransformerBlock(nn.Module):
    def __init__(self, in_channel_size, mid_channel_size, coords_channel_size, k):
        super().__init__()

        self.linear1 = nn.Conv1d(in_channel_size, mid_channel_size, 1)
        self.linear2 = nn.Conv1d(mid_channel_size, in_channel_size, 1)

        self.point_transforme = PointTransformerLayer(
            mid_channel_size, mid_channel_size, coords_channel_size, k)

    def forward(self, x, coords):
        identity = x
        x = self.linear1(x)
        x = self.point_transforme(x, coords)
        y = self.linear2(x) + identity
        return y

class TransitionDown(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, k, num_samples):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel_size, out_channel_size, (1,1)),
            nn.BatchNorm2d(out_channel_size),
            nn.ReLU(inplace=True)
        )

        self.k = k
        self.num_samples = num_samples

    def forward(self, x, coords):
        # Get p and x of fps indices
        fps_indices = furthest_point_sampling(coords, self.num_samples)
        # fps_x = index2points(x, fps_indices)
        fps_coords = index2points(coords, fps_indices) # p

        # Get knn indices
        knn_indices, _ = k_nearest_neighbors(fps_coords, coords, self.k)
        knn_x = index2points(x, knn_indices)

        # MLP
        knn_mlp_x = self.mlp(knn_x)

        # Use local max pooling. 
        y, _ = torch.max(knn_mlp_x, dim=-1)

        return y, fps_coords

# class TransitionUp(nn.Module):
#     def __init__(self, in_channel_size, out_channel_size):
#         super().__init__()

#         self.mlp_1 = nn.Sequential(
#             nn.Conv2d(in_channel_size, out_channel_size, (1,1)),
#             nn.BatchNorm2d(out_channel_size),
#             nn.ReLU(inplace=True)
#         )

#         self.mlp_2 = nn.Sequential(
#             nn.Conv2d(in_channel_size, out_channel_size, (1,1)),
#             nn.BatchNorm2d(out_channel_size),
#             nn.ReLU(inplace=True)
#         )

        






