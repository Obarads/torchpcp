import torch
from torch_cluster import fps

from torch_point_cloud.modules.functional import ball_query

def batch_fps(xyz:torch.tensor, num_samples:int):
    # How to use fps: examples/PointNet2ASIS/tests/fps_test.py
    device = xyz.device
    B, N, C = xyz.shape
    xyz = xyz.view(B*N, C)
    batch = torch.arange(0,B,dtype=torch.long, device=device)
    batch = batch.view(-1,1).repeat(1,N).view(B*N)
    fps_idx = fps(xyz, batch, num_samples/N, True)
    fps_idx = fps_idx.view(B, num_samples)
    idx_base = torch.arange(0, B, device=device).view(-1, 1)*N
    fps_idx = fps_idx - idx_base
    return fps_idx

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def pairwise_distance(xyz):
    inner = -2*torch.matmul(xyz.transpose(2, 1), xyz)
    xyz2 = torch.sum(xyz**2, dim=1, keepdim=True)
    return -xyz2 - inner - xyz2.transpose(2, 1)

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def knn(xyz, k, memory_saving=False):
    """
    args:
        xyz (Batch, channels, num_points): coordinate
        k int: k of kNN
        memory_saving bool: for gpu memory saving
    """
    if memory_saving:
        point_pairwise_distance_list = []
        for i in range(len(xyz)):
            point_pairwise_distance_list.append(pairwise_distance(xyz[i:i+1]))
        point_pairwise_distance = torch.cat(point_pairwise_distance_list, dim=0)
    else:
        point_pairwise_distance = pairwise_distance(xyz)
    return point_pairwise_distance.topk(k, dim=-1)[1] # (batch_size, num_points, k)

def index2points(points, idx):
    """Construct edge feature for each point
    Args:
        points: (batch_size, num_points, num_dims)
        nn_idx: (batch_size, num_points) or (batch_size, num_points, k)

    Returns:
        edge_features: (batch_size, num_points, num_dims) or (batch_size, num_points, k, num_dims)
    """
    B, N, C = points.shape # B, C, N = points.shape
    idx_shape = idx.shape # k = idx.shape[2]

    idx_base = torch.arange(0, B, device=points.device).view(-1, *[1]*(len(idx_shape)-1)) * N # if len(idx_shape) = 3, .view(-1, 1, 1)
    idx = idx + idx_base
    idx = idx.view(-1)

    feature = points.view(B*N, -1)
    feature = feature[idx, :]
    edge_feature = feature.view(*idx_shape, C)

    return edge_feature

