import os, sys
import numpy as np
from plyfile import PlyData, PlyElement

from torch_point_cloud.utils2 import converter

##
## Write 
##

def write_pc(filename, xyz, rgb=None):
    """
    write into a ply file
    ref.:https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/provider.py
    """
    if rgb is None:
        # len(xyz[0]): for a xyz list, I don't use `.shape`.
        rgb = np.full((len(xyz), 3), 255, dtype=np.int32)
    if not isinstance(xyz, (np.ndarray, np.generic)):
        xyz = np.array(xyz, np.float32)

    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def write_pc_embedding(filename, xyz, embeddings):
    """
    write a ply with colors corresponding to geometric features
    ref.:https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/provider.py
    """
    color = converter.embedding_to_color(embeddings)
    write_pc(filename, xyz, color)

def write_pc_label(filename, xyz, labels, seed=0, color_map:list=None):
    point_colors = converter.label_to_color(labels, seed=seed, color_map=color_map)
    write_pc(filename, xyz, point_colors)

def check_label_colors(file_name, num_points, seed=0, margin=0.5):
    """
    Output label colors of write_label.
    """
    xyz, labels = converter.check_label_colors(num_points, seed=seed, margin=margin)
    write_pc_label(file_name, xyz, labels, seed=seed)

def write_pc_intensity(file_name, xyz, intensity):
    rgb = converter.intensity_to_color(xyz, intensity)
    write_pc(file_name,xyz,rgb)


