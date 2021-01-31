# import math

# import torch
# from torch import nn

# # https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/models/blocks.py
# class KPConv(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size):
#         super().__init__()

#         self.weights = nn.Parameter(
#             torch.zeros(kernel_size, in_channel, out_channel, dtype=torch.float32),
#             requires_grad=True
#         )

#         # Reset parameters.
#         self.reset_parameters()

#         # Get initialized kernel points.
#         self.kernel_points = self.init_KP()

#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.kernel_size = self.kernel_size

#     def reset_parameters(self):
#         """
#         Reset self.weights with kaiming Initializations.
#         """
#         nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

#     def init_KP(self):
#         """
#         Initialize the kernel point positions in a sphere.

#         Return 
#         ------
#         KP_Parameters : nn.Parameter
#             the tensor of kernel points
#         """
    
#         # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
#         K_points_numpy = load_kernels(self.radius,
#                                       self.K,
#                                       dimension=self.p_dim,
#                                       fixed=self.fixed_kernel_points)

#         return nn.Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
#                             requires_grad=False)

#     def forward(self, q_pts, s_pts, neighb_inds, x):

#         # Add a fake point in the last row for shadow neighbors
#         s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

#         # Get neighbor points [n_points, n_neighbors, dim]
#         neighbors = s_pts[neighb_inds, :]

#         # Center every neighborhood
#         neighbors = neighbors - q_pts.unsqueeze(1)

#         deformed_K_points = self.kernel_points

#         # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
#         neighbors.unsqueeze_(2)
#         differences = neighbors - deformed_K_points

#         # Get the square distances [n_points, n_neighbors, n_kpoints]
#         sq_distances = torch.sum(differences ** 2, dim=3)

#         new_neighb_inds = neighb_inds

#         # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
#         if self.KP_influence == 'constant':
#             # Every point get an influence of 1.
#             all_weights = torch.ones_like(sq_distances)
#             all_weights = torch.transpose(all_weights, 1, 2)

#         elif self.KP_influence == 'linear':
#             # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
#             all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
#             all_weights = torch.transpose(all_weights, 1, 2)

#         elif self.KP_influence == 'gaussian':
#             # Influence in gaussian of the distance.
#             sigma = self.KP_extent * 0.3
#             all_weights = radius_gaussian(sq_distances, sigma)
#             all_weights = torch.transpose(all_weights, 1, 2)
#         else:
#             raise ValueError('Unknown influence function type (config.KP_influence)')

#         # In case of closest mode, only the closest KP can influence each point
#         if self.aggregation_mode == 'closest':
#             neighbors_1nn = torch.argmin(sq_distances, dim=2)
#             all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

#         elif self.aggregation_mode != 'sum':
#             raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

#         # Add a zero feature for shadow neighbors
#         x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

#         # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
#         neighb_x = gather(x, new_neighb_inds)

#         # Apply distance weights [n_points, n_kpoints, in_fdim]
#         weighted_features = torch.matmul(all_weights, neighb_x)

#         # Apply network weights [n_kpoints, n_points, out_fdim]
#         weighted_features = weighted_features.permute((1, 0, 2))
#         kernel_outputs = torch.matmul(weighted_features, self.weights)

#         # Convolution sum [n_points, out_fdim]
#         return torch.sum(kernel_outputs, dim=0)

