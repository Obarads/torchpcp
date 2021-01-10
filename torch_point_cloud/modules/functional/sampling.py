import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Function

from .torch_c.backend import _backend

def furthest_point_sample(coords, num_samples):
    """
    Uses iterative furthest point sampling.

    Parameters
    ----------
    coords :  torch.tensor [B, 3, N]
        coordinates of points
    num_samples : int 
        number of sampeld centers

    Return
    ------
    indices : torch.tensor [B, M]
        indexes of sampled centers
    """
    coords = coords.contiguous()
    indices = _backend.furthest_point_sampling(coords, num_samples)
    # return gather(coords, indices)
    return indices

def index2points(points, idx):
    """Construct edge feature for each point

    Parameters
    ----------
    points: (batch_size, num_dims, num_points)
    idx: (batch_size, num_points) or (batch_size, num_points, k)

    Returns
    -------
    edge_features: (batch_size, num_dims, num_points) or (batch_size, num_dims, num_points, k)
    """

    B, C, N = points.shape
    idx_shape = idx.shape

    idx_base = torch.arange(0, B, device=points.device).view(-1, *[1]*(len(idx_shape)-1)) * N # if len(idx_shape) = 3, .view(-1, 1, 1)
    idx = idx + idx_base
    idx = idx.view(-1)

    # (batch_size, num_dims, num_points) -> (batch_size, num_points, num_dims)
    points = points.transpose(2, 1).contiguous()

    feature = points.view(B*N, -1)
    feature = feature[idx, :]
    edge_feature = feature.view(*idx_shape, C)

    # (batch_size, num_points, num_dims) -> (batch_size, num_dims, num_points, ...)
    edge_feature = edge_feature.permute(0, -1, *range(1, len(idx_shape)))
    edge_feature = edge_feature.contiguous()

    return edge_feature

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def pairwise_distance(xyz1, xyz2):
    """
    Compute distances between xyz1 and xyz2.

    Parameters
    ----------
    xyz1 : torch.tensor [B, C, N]
        xyz tensor
    xyz2 : torch.tensor [B, C, M]
        xyz tensor
    
    Return
    ------
    distances : torch.tesnor [B, N, M]
        distances between xyz1 and xyz2
    """
    inner = -2*torch.matmul(xyz1.transpose(2, 1), xyz2)
    xyz_column = torch.sum(xyz2**2, dim=1, keepdim=True)
    xyz_row = torch.sum(xyz1**2, dim=1, keepdim=True).transpose(2,1)
    return -xyz_column - inner - xyz_row

def k_nearest_neighbors(center_coords, coords, k):
    """
    Compute k nearest neighbors between coords and center_coords.

    Parameters
    ----------
    center_coords : torch.tensor [B, C, M]
        xyz of center points
    coords : torch.tensor [B, C, N]
        xyz of all points
    k : int
        number of nearest neighbors

    Return
    ------
    idxs: torch.tesnor [B, M, k]
        top k idx between coords and center_coords
    distance : torch.tesnor [B, M, k]
        top k distances between coords and center_coords
    """
    coords = coords.contiguous()
    center_coords = center_coords.contiguous()
    idxs, dists = _backend.k_nearest_neighbors(coords, center_coords, k)
    return idxs, dists

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def py_k_nearest_neighbors(xyz1, xyz2, k, memory_saving=False):
    """
    Compute k nearest neighbors between xyz1 and xyz2.

    Parameters
    ----------
    xyz1 : torch.tensor [B, C, N]
        xyz of center points
    xyz2 : torch.tensor [B, C, M]
        xyz of all points
    k : int
        number of nearest neighbors
    memory_saving : bool
        memory saving with "for"

    Return
    ------
    idxs: torch.tesnor [B, N, k]
        top k idx between xyz1 and xyz2
    distance : torch.tesnor [B, N, k]
        top k distances between xyz1 and xyz2
    """
    B, _, _ = xyz1.shape
    if memory_saving:
        point_pairwise_distance_list = []
        for i in range(B):
            point_pairwise_distance_list.append(
                pairwise_distance(xyz1[i:i+1], xyz2[i:i+1])
            )
        point_pairwise_distances = torch.cat(point_pairwise_distance_list, dim=0)
    else:
        point_pairwise_distances = pairwise_distance(xyz1, xyz2)
    top_dists, idxs = point_pairwise_distances.topk(k, dim=-1) 
    top_dists = -1 * top_dists
    return  idxs, top_dists

def localize(center_points, points):
    """
    Parameters
    ----------
    center_points: [B, C, N]
    points: [B, C, N, k]
    """
    B, C, N, k = points.shape
    center_points = center_points.view(B, C, N, 1).repeat(1, 1, 1, k).contiguous()
    local_points = points - center_points
    return local_points

def random_sampling(num_points, num_samples):
    """
    get indices using random subsampling

    Parameters
    ----------
    num_points : int
        number of points
    num_samples : int
        number of samples

    Return
    ------
    sampled_points_indices
        subsampled point indices
    """
    sampled_points_indices = np.random.choice(num_points, num_samples, replace=False)
    return sampled_points_indices

