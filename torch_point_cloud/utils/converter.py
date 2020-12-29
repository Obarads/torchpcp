import numpy as np
from typing import List
from sklearn.decomposition import PCA

##
## label
##

def sparseLabel_to_denseLabelConverter(label):
    unique_label = np.unique(label)
    converted_num = np.arange(0,len(unique_label))
    label_converter = np.full(np.max(unique_label)+1,-1)
    label_converter[unique_label] = converted_num
    return label_converter

def sparseLabel_to_denseLabel(label):
    converter = sparseLabel_to_denseLabelConverter(label)
    converted_label = list(map(lambda x:converter[x],label))
    return converted_label

def batch_sparseLabel_to_denseLabel(labels):
    return np.apply_along_axis(sparseLabel_to_denseLabel, 1, labels)

def label_to_color(labels, seed=0, color_map:list=None):
    np.random.seed(seed)
    max_labels = np.max(labels)
    if color_map is None:
        label_color_list = np.random.randint(0,255,(max_labels+1,3))
    else:
        label_color_list = color_map
    point_colors = np.array([ np.array([0,0,0]) if 0 > i else label_color_list[i] for i in labels])
    return point_colors

##
## check label colors
##

def check_label_colors(num_points, seed=0, margin=0.5):
    """
    Output label colors of write_label.
    """
    xyz = np.zeros((num_points, 3))
    margin_coord = np.arange(num_points) * margin
    xyz[:,0] = margin_coord
    labels = np.arange(num_points)
    return xyz, labels

##
## mask
##

def label_to_mask(labels, max_label=None):
    # N = labels.shape
    N = len(labels)
    if max_label is None:
        max_label = np.amax(labels)
    mask = np.zeros((N,max_label+1))
    mask[np.arange(N), labels] = 1
    return mask

##
## intensity
##

def intensity_to_color(xyz, intensity) -> np.ndarray:
    intensity = intensity[:,np.newaxis]
    intensity = np.concatenate([intensity,intensity,intensity],axis=1)
    rgb = (intensity*255)
    rgb = rgb.astype(np.int32)
    return rgb

##
## embeddings
##

def embedding_to_color(embeddings):
    """
    write a ply with colors corresponding to geometric features
    ref.:https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/provider.py
    """
    if embeddings.shape[1]>3:
        pca = PCA(n_components=3)
        #pca.fit(np.eye(embeddings.shape[1]))
        pca.fit(np.vstack((np.zeros((embeddings.shape[1],)),np.eye(embeddings.shape[1]))))
        embeddings = pca.transform(embeddings)
    value = np.minimum(np.maximum((embeddings+1)/2,0),1)
    color = np.array(255 * value, dtype='uint8')
    return color


# def mesh_to_points(vertices, faces):
#     """
#     ref: https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp
#     """
#     SAMPLE_POINTS_:int = default_number_samples
#     leaf_size:float = default_leaf_size
#     vis_result:bool = vis_result
#     write_normals:bool = write_normals
#     write_colors:bool = write_colors

#     # Parse the command line arguments for .ply and PCD files
#     pcd_file_indices:List[int] = pcd_file_indices
#     if len(pcd_file_indices) != 1:
#         print("Need a single output PCD file to continue.")
#         return -1
#     ply_file_indices:List[int] = ply_file_indices
#     obj_file_indices:List[int] = obj_file_indices
#     if len(ply_file_indices) != 1 and len(obj_file_indices) != 1:
#         print("Need a single input PLY/OBJ file to continue.\n");
#         return -1
#     return 


