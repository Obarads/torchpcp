import numpy as np

from .torch_c.backend import _backend

def furthest_point_sampling(coords, num_samples):
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

