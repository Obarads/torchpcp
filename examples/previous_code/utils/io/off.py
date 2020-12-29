import os, sys
import numpy as np
# from plyfile import PlyData, PlyElement
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

def write_pc_label(filename, xyz, labels, seed=0, color_map:list=None):
    np.random.seed(seed)
    max_labels = np.max(labels)
    if color_map is None:
        label_color_list = np.random.randint(0,255,(max_labels+1,3))
    else:
        label_color_list = color_map
    point_colors = np.array([ np.array([0,0,0]) if 0 > i else label_color_list[i] for i in labels])
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

def read(file):
    """
    ref: https://davidstutz.de/visualizing-triangular-meshes-from-off-files-using-python-occmodel/
    Reads vertices and faces from an off file.
 
    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file)

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]
 
        assert lines[0] == 'OFF'
 
        parts = lines[1].split(' ')
        assert len(parts) == 3
 
        num_vertices = int(parts[0])
        assert num_vertices > 0
 
        num_faces = int(parts[1])
        assert num_faces > 0
 
        vertices = []
        for i in range(num_vertices):
            vertex = lines[2 + i].split(' ')
            vertex = [float(point) for point in vertex]
            assert len(vertex) == 3
 
            vertices.append(vertex)
 
        faces = []
        for i in range(num_faces):
            face = lines[2 + num_vertices + i].split(' ')
            face = [int(index) for index in face]
 
            assert face[0] == len(face) - 1
            for index in face:
                assert index >= 0 and index < num_vertices
 
            assert len(face) > 1
 
            faces.append(face)
 
        return vertices, faces


