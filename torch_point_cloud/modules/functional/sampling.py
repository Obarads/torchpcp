import numpy as np
import torch
from torch.autograd import Function

from .torch_c.backend import _backend

# __all__ = ['gather', 'furthest_point_sample', 'logits_mask']

# https://github.com/mit-han-lab/pvcnn/blob/db13331a46f672e74e7b5bde60e7bf30d445cd2d/modules/functional/sampling.py#L10
class Gather(Function):
    @staticmethod
    def forward(ctx, features, indices):
        """
        Gather
        :param ctx:
        :param features: features of points, FloatTensor[B, C, N]
        :param indices: centers' indices in points, IntTensor[b, m]
        :return:
            centers_coords: coordinates of sampled centers, FloatTensor[B, C, M]
        """
        features = features.contiguous()
        indices = indices.int().contiguous()
        ctx.save_for_backward(indices)
        ctx.num_points = features.size(-1)
        return _backend.gather_features_forward(features, indices)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_features = _backend.gather_features_backward(grad_output.contiguous(), indices, ctx.num_points)
        return grad_features, None

gather = Gather.apply

# https://github.com/mit-han-lab/pvcnn/blob/db13331a46f672e74e7b5bde60e7bf30d445cd2d/modules/functional/sampling.py#L51
def logits_mask(coords, logits, num_points_per_object):
    """
    Use logits to sample points
    :param coords: coords of points, FloatTensor[B, 3, N]
    :param logits: binary classification logits, FloatTensor[B, 2, N]
    :param num_points_per_object: M, #points per object after masking, int
    :return:
        selected_coords: FloatTensor[B, 3, M]
        masked_coords_mean: mean coords of selected points, FloatTensor[B, 3]
        mask: mask to select points, BoolTensor[B, N]
    """
    batch_size, _, num_points = coords.shape
    mask = torch.lt(logits[:, 0, :], logits[:, 1, :])   # [B, N]
    num_candidates = torch.sum(mask, dim=-1, keepdim=True)  # [B, 1]
    masked_coords = coords * mask.view(batch_size, 1, num_points)  # [B, C, N]
    masked_coords_mean = torch.sum(masked_coords, dim=-1) / torch.max(num_candidates,
                                                                      torch.ones_like(num_candidates)).float()  # [B, C]
    selected_indices = torch.zeros((batch_size, num_points_per_object), device=coords.device, dtype=torch.int32)
    for i in range(batch_size):
        current_mask = mask[i]  # [N]
        current_candidates = current_mask.nonzero().view(-1)
        current_num_candidates = current_candidates.numel()
        if current_num_candidates >= num_points_per_object:
            choices = np.random.choice(current_num_candidates, num_points_per_object, replace=False)
            selected_indices[i] = current_candidates[choices]
        elif current_num_candidates > 0:
            choices = np.concatenate([
                np.arange(current_num_candidates).repeat(num_points_per_object // current_num_candidates),
                np.random.choice(current_num_candidates, num_points_per_object % current_num_candidates, replace=False)
            ])
            np.random.shuffle(choices)
            selected_indices[i] = current_candidates[choices]
    selected_coords = gather(masked_coords - masked_coords_mean.view(batch_size, -1, 1), selected_indices)
    return selected_coords, masked_coords_mean, mask

# https://github.com/mit-han-lab/pvcnn/blob/db13331a46f672e74e7b5bde60e7bf30d445cd2d/modules/functional/sampling.py#L37
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

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def k_nearest_neighbors(xyz1, xyz2, k, memory_saving=False):
    """
    Compute k nearest neighbors between xyz1 and xyz2.

    Parameters
    ----------
    xyz1 : torch.tensor [B, C, N]
        xyz tensor
    xyz2 : torch.tensor [B, C, M]
        xyz tensor
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
