from torch.autograd import Function

from .torch_c.backend import _backend

__all__ = ['ball_query']

# https://github.com/mit-han-lab/pvcnn/blob/master/modules/functional/ball_query.py
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
        centers_coords = centers_coords.contiguous()
        points_coords = points_coords.contiguous()
        return _backend.ball_query(centers_coords, points_coords, radius, num_neighbors)
