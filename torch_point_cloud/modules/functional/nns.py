import numpy as np
import random

import torch

from .torch_c.backend import _backend

"""
TOC

Other function
K nearest neighbors
Ball query

"""


##
# Other function
##

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
    # base: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    inner = -2*torch.matmul(xyz1.transpose(2, 1), xyz2)
    xyz_column = torch.sum(xyz2**2, dim=1, keepdim=True)
    xyz_row = torch.sum(xyz1**2, dim=1, keepdim=True).transpose(2, 1)
    return -xyz_column - inner - xyz_row

###
# K nearest neighbors
###

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
        point_pairwise_distances = torch.cat(
            point_pairwise_distance_list, dim=0)
    else:
        point_pairwise_distances = pairwise_distance(xyz1, xyz2)
    top_dists, idxs = point_pairwise_distances.topk(k, dim=-1)
    dists = -1 * top_dists
    return idxs, dists


##
# Ball query
##

def ball_query(centers_coords, points_coords, radius, num_neighbors):
    """
    Ball query

    Parameters
    ----------
    centers_coords : torch.tensor [B, 3, M]
        coordinates of centers
    points_coords : torch.tensor [B, 3, N]
        coordinates of points
    radius : float
        radius of ball query
    num_neighbors : int
        maximum number of neighbors

    Return
    ------
    neighbor_indices : torch.tensor [B, M, U]
        indices of neighbors
    """
    # https://github.com/mit-han-lab/pvcnn/blob/master/modules/functional/ball_query.py
    centers_coords = centers_coords.contiguous()
    points_coords = points_coords.contiguous()
    return _backend.ball_query(centers_coords, points_coords, radius, num_neighbors)

