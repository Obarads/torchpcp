import torch
from torch_cluster import fps

# from torch_point_cloud.models.modules.ball_query import BallQuery
from torch_point_cloud.modules.functional.ball_query import ball_query

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L19
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L43
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L63
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # xyz_type = xyz.dtype
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L87
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device)
    group_idx = group_idx.view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

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

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L110
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3] -> [B, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    fps_idx = batch_fps(xyz, npoint)
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = ball_query(new_xyz.transpose(1,2), xyz.transpose(1,2), radius, nsample) # add
    idx = idx.type(torch.long) # add
    # idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_util.py#L146
def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def pairwise_distance(xyz):
    inner = -2*torch.matmul(xyz.transpose(2, 1), xyz)
    xyz2 = torch.sum(xyz**2, dim=1, keepdim=True)
    return -xyz2 - inner - xyz2.transpose(2, 1)

# https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def knn_index(xyz, k, memory_saving=False):
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
        points: (batch_size, num_dims, num_points)
        nn_idx: (batch_size, num_points, k)
        k: int

    Returns:
        edge_features: (batch_size, num_dims, num_points, k)
    """
    B, C, N = points.shape
    k = idx.shape[2]
    idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1)*N
    idx = idx + idx_base
    idx = idx.view(-1)
    points = points.transpose(2, 1).contiguous() # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = points.view(B*N, -1)
    feature = feature[idx, :]
    edge_feature = feature.view(B, N, k, C)
    edge_feature = edge_feature.transpose(1,3).transpose(2,3)

    return edge_feature