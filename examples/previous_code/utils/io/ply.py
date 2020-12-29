import os, sys
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.decomposition import PCA

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
    if embeddings.shape[1]>3:
        pca = PCA(n_components=3)
        #pca.fit(np.eye(embeddings.shape[1]))
        pca.fit(np.vstack((np.zeros((embeddings.shape[1],)),np.eye(embeddings.shape[1]))))
        embeddings = pca.transform(embeddings)
        
    #value = (embeddings-embeddings.mean(axis=0))/(2*embeddings.std())+0.5
    #value = np.minimum(np.maximum(value,0),1)
    #value = (embeddings)/(3 * embeddings.std())+0.5
    value = np.minimum(np.maximum((embeddings+1)/2,0),1)

    color = np.array(255 * value, dtype='uint8')
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i in range(0, 3):
        vertex_all[prop[i][0]] = xyz[:, i]
    for i in range(0, 3):
        vertex_all[prop[i+3][0]] = color[:, i]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    
    ply.write(filename)

def label_to_color(labels, seed=0, color_map:list=None):
    np.random.seed(seed)
    max_labels = np.max(labels)
    if color_map is None:
        label_color_list = np.random.randint(0,255,(max_labels+1,3))
    else:
        label_color_list = color_map
    point_colors = np.array([ np.array([0,0,0]) if 0 > i else label_color_list[i] for i in labels])
    return point_colors

def write_pc_label(filename, xyz, labels, seed=0, color_map:list=None):
    point_colors = label_to_color(labels, seed=seed, color_map=color_map)
    write_pc(filename, xyz, point_colors)

def check_label_colors(file_name, num_points, seed=0, margin=0.5):
    """
    Output label colors of write_label.
    """
    xyz = np.zeros((num_points, 3))
    margin_coord = np.arange(num_points) * margin
    xyz[:,0] = margin_coord
    labels = np.arange(num_points)
    write_pc_label(file_name, xyz, labels, seed=seed)

def write_pc_intensity(file_name, xyz, intensity):
    intensity = intensity[:,np.newaxis]
    intensity = np.concatenate([intensity,intensity,intensity],axis=1)
    rgb = (intensity*255)
    rgb = rgb.astype(np.int32)
    write_pc(file_name,xyz,rgb)

##
## Read
##



