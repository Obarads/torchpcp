import torch
from torch import nn
from torch.nn import functional as F

from torchpcp.modules.functional.nns import k_nearest_neighbors
from torchpcp.modules.functional.other import index2points, localize
from torchpcp.modules.functional.sampling import furthest_point_sampling
from torchpcp.modules.Layer import PointwiseConv2D, PointwiseConv1D

# from einops import repeat

from torchpcp.utils.pytorch_tools import t2n
from torchpcp.utils.io.ply import write_pc

class PointTransformerLayer(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, coords_channel_size, k):
        super().__init__()
        self.input_linear = nn.Conv1d(in_channel_size, out_channel_size*3, 1, bias=False)

        #  gamma (mapping function)
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(out_channel_size, out_channel_size, (1,1), bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channel_size, out_channel_size, (1,1), bias=False),
            nn.ReLU(inplace=True),
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
        coords: torch.tensor (B, 3, N)
        """

        # Get knn indexes.
        knn_indices, _ = k_nearest_neighbors(coords, coords, self.k) # get (B, N, k)

        # Get delta.
        outputs_delta = self.pe_delta(coords, knn_indices) # get (B, C, N, k)

        # Get pointwise feature.
        outputs_phi, outputs_psi, outputs_alpha = torch.chunk(
            self.input_linear(features), chunks=3, dim=1) # to (B, C, N) x 3

        # Get weights.
        outputs_psi = index2points(outputs_psi, knn_indices) # to (B, C, N, k)
        inputs_gamma = localize(outputs_phi, outputs_psi) * -1 + outputs_delta # get (B, C, N, k)
        outputs_gamma = self.mlp_gamma(inputs_gamma)
        outputs_rho = self.normalization_rho(outputs_gamma)

        # \alpha(x_j) + \delta
        outputs_alpha = index2points(outputs_alpha, knn_indices) # to (B, C, N, k)
        outputs_alpha_delta = outputs_alpha + outputs_delta
        # outputs_alpha_delta = outputs_alpha

        # compute value with hadamard product and aggregation
        outputs_hp = outputs_rho * outputs_alpha_delta
        outputs_aggregation = torch.sum(outputs_hp, dim=-1)  # get (B, C, N)

        return outputs_aggregation

class PositionEncoding(nn.Module):
    def __init__(self, in_channel_size, out_channel_size):
        super().__init__()
        # theta (encoding function)
        self.mlp_theta = nn.Sequential(
            nn.Conv2d(in_channel_size, out_channel_size, (1,1), bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv2d(out_channel_size, out_channel_size, (1,1), bias=False),
            nn.ReLU(inplace=True),
        )
        # self.k = k

    def forward(self, coords, knn_indices):
        """
        Parameters
        ----------
        coords: torch.tensor (B, C, N)
        """

        # Get spaces between points.
        knn_coords = index2points(coords, knn_indices)
        coords_space = localize(coords, knn_coords) * -1

        # Use theta.
        outputs = self.mlp_theta(coords_space)

        return outputs

class PointTransformerBlock(nn.Module):
    def __init__(self, in_channel_size, mid_channel_size, coords_channel_size, k):
        super().__init__()

        self.linear1 = nn.Conv1d(in_channel_size, mid_channel_size, 1, bias=False)
        self.linear2 = nn.Conv1d(mid_channel_size, in_channel_size, 1, bias=False)

        self.point_transforme = PointTransformerLayer(
            mid_channel_size, mid_channel_size, coords_channel_size, k)

    def forward(self, x, coords):
        identity = x
        x = self.linear1(x)
        x = self.point_transforme(x, coords)
        x = self.linear2(x)
        y = x + identity
        return y

class TransitionDown(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, k, num_samples):
        super().__init__()
        self.mlp = PointwiseConv2D(in_channel_size, out_channel_size, 
                                   conv_args={"bias": False})

        self.k = k
        self.num_samples = num_samples

    def forward(self, x, coords):
        # Get p and x of fps indices
        fps_indices = furthest_point_sampling(coords, self.num_samples)
        fps_coords = index2points(coords, fps_indices) # p

        # Get knn indices
        knn_indices, _ = k_nearest_neighbors(fps_coords, coords, self.k)
        knn_x = index2points(x, knn_indices)

        # MLP
        knn_mlp_x = self.mlp(knn_x)

        # Use local max pooling. 
        y, _ = torch.max(knn_mlp_x, dim=-1)

        return y, fps_coords

class NonTrasition(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, k):
        super().__init__()
        self.mlp = PointwiseConv2D(in_channel_size, out_channel_size, 
                                   conv_args={"bias": False})
        self.k = k

    def forward(self, x, coords):
        # Get knn indices
        knn_indices, _ = k_nearest_neighbors(coords, coords, self.k)
        knn_x = index2points(x, knn_indices)

        # MLP
        knn_mlp_x = self.mlp(knn_x)

        # Use local max pooling.
        y, _ = torch.max(knn_mlp_x, dim=-1)

        return y, coords



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

        






