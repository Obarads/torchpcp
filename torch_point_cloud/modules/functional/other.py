import numpy as np
import random

import torch
from torch import nn
from torch.autograd import Function

from .torchpct_cpp.backend import _backend

def gather(point_clouds, indices):
    output_shape = [*point_clouds.shape[:2],*indices.shape[1:]]
    point_idxs_size = sum(output_shape[2:])
    gathered_point_clouds = _backend.gather(
        point_idxs_size,
        output_shape,
        point_clouds, 
        indices
    )
    return gathered_point_clouds

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


def index2row(idx):
    """
    Convert (B,N,...) to (B*N*...) and add B*N into indexs

    Parameters
    ----------
        idx: (batch_size, num_points, ...)
    Returns
    -------
        one_row: (batch_size*num_dims*...)
    """
    idx_shape = idx.shape
    B, N = idx_shape[:2]

    idx_base = torch.arange(0, B, device=idx.device).view(-1, *[1]*(len(idx_shape)-1)) * N # if len(idx_shape) = 3, .view(-1, 1, 1)
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx

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


