import torch
from torch import nn
from torch.nn import functional as F

from torch_point_cloud.modules.Layer import MLP2D
from torch_point_cloud.modules.functional.sampling import (
    index2points,
    furthest_point_sample
)
from torch_point_cloud.modules.functional.ball_query import ball_query

def sampling_layer(coords, num_samples):
    """
    Sampling layer in PointNet++

    Parameters
    ----------
    coords : torch.tensor [B, 3, N]
        xyz tensor
    num_samples : int
       number of samples for furthest point sample

    Return
    ------
    sampled_coords : torch.tensor [B, 3, num_samples]
        sampled xyz using furthest point sample
    """
    fps_idx = furthest_point_sample(coords, num_samples) # fps_idx = batch_fps(coords, num_samples)
    fps_idx = fps_idx.type(torch.long)
    sampled_coords = index2points(coords, fps_idx)
    return sampled_coords

def group_layer(coords, center_coords, num_samples, radius, points=None):
    """
    Group layer in PointNet++

    Parameters
    ----------
    coords : torch.tensor [B, 3, N]
        xyz tensor
    center_coords : torch.tensor [B, 3, N']
        xyz tensor of ball query centers
    num_samples : int
       maximum number of samples for ball query
    radius : float
        radius of ball query
    points : torch.tensor [B, C, N]
        Concatenate points to return value.

    Return
    ------
    new_points : torch.tensor [B, 3, N', num_samples] or [B, 3+C, N', num_samples]
        If points is not None, new_points shape is [B, 3+C, N', num_samples].
    """
    # Get sampled coords idx by ball query.
    idx = ball_query(center_coords, coords, radius, num_samples)
    idx = idx.type(torch.long)

    # Convert idx to coords
    grouped_coords = index2points(coords, idx)
    center_coords = torch.unsqueeze(center_coords, 3)
    grouped_coords_norm = grouped_coords - center_coords

    if points is not None:
        grouped_points = index2points(points, idx)
        new_points = torch.cat([grouped_coords_norm, grouped_points], dim=1)
        # note: PointNetSetAbstractionMsg is different order of concatenation.
        # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/2d08fa40635cc5eafd14d19d18e3dc646171910d/models/pointnet_util.py#L253
    else:
        new_points = grouped_coords_norm
    
    return new_points

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L146
def sampling_and_group_layer_all(xyz, points=None):
    """
    Group layer (all)

    Parameters
    ----------
    xyz : torch.tensor [B, 3, N]
        xyz tensor
    points : torch.tensor [B, D, N]
        Concatenate points to return value (new_points).

    Returns
    -------
    new_xyz : torch.tensor [B, 3, 1]
        new xyz
    new_points : torch.tensor [B, 3, 1, N] or [B, 3+D, 1, N]
        If points is not None, new_points shape is [B, 3+D, 1, N].
    """
    device = xyz.device
    B, C, N = xyz.shape
    new_xyz = torch.zeros(B, C, 1).to(device)
    grouped_xyz = xyz.view(B, C, 1, N)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, -1, 1, N)], dim=1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L166
class PointNetSetAbstraction(nn.Module):
    """
    PointNetSetAbstraction

    Parameters
    ----------
    num_fps_points : int
        number of samples for furthest point sample
    radius : float
        radius of ball query
    num_bq_points : int
        maximum number of samples for ball query
    init_in_channel : int
        input channel size
    mlp : list [O]
        MLP output channel sizes of PointNet layer
    group_all : bool
        group_all

    See Also
    --------
    O : out channel sizes
    """

    def __init__(self, num_fps_points, radius, num_bq_points, init_in_channel, mlp, group_all):
        super().__init__()

        # create MLPs for a PointNet Layer
        layers = []
        in_channel = init_in_channel
        for out_channel in mlp:
            layers.append(MLP2D(in_channel, out_channel))
            in_channel = out_channel
        self.mlp = nn.Sequential(*layers)

        self.num_fps_points = num_fps_points
        self.radius = radius
        self.num_bq_points = num_bq_points
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Parameters
        ----------
        xyz : torch.tensor [B, 3, N]
            xyz tensor
        points : torch.tensor [B, C, N]
            features of points

        Returns
        -------
        new_xyz : torch.tensor [B, 3, num_fps_points]
            sampled xyz using furthest point sample
        new_points : torch.tensor [B, D, num_fps_points]
            features of points by PointNetSetAbstraction
        """

        # Sampling Layer and Grouping Layer
        if self.group_all:
            new_xyz, new_points = sampling_and_group_layer_all(xyz, points)
        else:
            new_xyz = sampling_layer(xyz, self.num_fps_points)
            new_points = group_layer(xyz, new_xyz, self.num_bq_points, 
                                     self.radius, points=points)

        # PointNet Layer
        new_points = self.mlp(new_points)
        new_points = torch.max(new_points, -1)[0]

        return new_xyz, new_points

class PointNetSetAbstractionMSG(nn.Module):
    """
    PointNetSetAbstractionMSG

    Parameters
    ----------
    num_fps_points : int
        number of samples for furthest point sample
    radius_list : list [G]
        radius of ball query
    num_bq_points_list : list [G]
        list of the maximum number of ball query samples
    init_in_channel : int
        input channel size
    mlp_list : list [G, O]
        list of MLP output channel sizes of PointNet layer

    See Also
    --------
    G : number of groups of MSG 

    O : out channel sizes
    """

    def __init__(self, num_fps_points, radius_list, num_bq_points_list, 
                 init_in_channel, mlp_list):
        super().__init__()
        
        self.mlp_list = nn.ModuleList()
        for mlp in mlp_list:
            layers = []
            in_channel = init_in_channel + 3
            for out_channel in mlp:
                layers.append(MLP2D(in_channel, out_channel))
                in_channel = out_channel
            self.mlp_list.append(nn.Sequential(*layers))

        self.num_fps_points = num_fps_points
        self.radius_list = radius_list
        self.num_bq_points_list = num_bq_points_list

    # https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L210
    def forward(self, xyz, points):
        # B, C, N = xyz.shape

        new_xyz = sampling_layer(xyz, self.num_fps_points)

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.num_bq_points_list[i]
            grouped_points = group_layer(xyz, new_xyz, K, radius)

            # shape: [B, D, K, S]
            grouped_points = grouped_points.permute(0, 3, 2, 1)

            # PointNet layer
            grouped_points = self.mlp_list[i](grouped_points)
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        # MSG Concat
        new_points_concat = torch.cat(new_points_list, dim=1)

        return new_xyz, new_points_concat

