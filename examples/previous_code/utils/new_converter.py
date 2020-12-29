import numpy as np
from typing import List

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

def label_to_mask(labels, max_label=None):
    # N = labels.shape
    N = len(labels)
    if max_label is None:
        max_label = np.amax(labels)
    mask = np.zeros((N,max_label+1))
    mask[np.arange(N), labels] = 1
    return mask

def mesh_to_points(vertices, faces):
    """
    ref: https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp
    """
    SAMPLE_POINTS_:int = default_number_samples
    leaf_size:float = default_leaf_size
    vis_result:bool = vis_result
    write_normals:bool = write_normals
    write_colors:bool = write_colors

    # Parse the command line arguments for .ply and PCD files
    pcd_file_indices:List[int] = pcd_file_indices
    if len(pcd_file_indices) != 1:
        print("Need a single output PCD file to continue.")
        return -1
    ply_file_indices:List[int] = ply_file_indices
    obj_file_indices:List[int] = obj_file_indices
    if len(ply_file_indices) != 1 and len(obj_file_indices) != 1:
        print("Need a single input PLY/OBJ file to continue.\n");
        return -1

    

    return 


