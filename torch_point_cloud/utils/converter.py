import os, sys

sys.path.append("./")

import numpy as np
from plyfile import PlyData, PlyElement
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import torch

###
### Original:  https://github.com/loicland/superpoint_graph/
###
def write_ply(filename, xyz, rgb=None):
    """
    write into a ply file
    ref.:https://github.com/loicland/superpoint_graph/blob/ssp%2Bspg/partition/provider.py
    """
    if rgb is None:
        # len(xyz[0]): for a xyz list, I don't use `.shape`.
        rgb = np.full((len(xyz), 3), 255, dtype=np.int32)
    prop = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_all = np.empty(len(xyz), dtype=prop)
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop][0]] = xyz[:, i_prop]
    for i_prop in range(0, 3):
        vertex_all[prop[i_prop+3][0]] = rgb[:, i_prop]
    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=True)
    ply.write(filename)

def embedding2ply(filename, xyz, embeddings):
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
###
### Original: END
###
try:
    from tsnecuda import TSNE
except:
    from sklearn.manifold import TSNE

def write_tsne(label_dict, embedding, extension="png", tsne_model=None):
    """
    base:https://medium.com/analytics-vidhya/super-fast-tsne-cuda-on-kaggle-b66dcdc4a5a4
    """
    if tsne_model is None:
        tsne_model = TSNE()

    x_embedding = tsne_model.fit_transform(embedding)

    for key in label_dict:
        label = label_dict[key]
        embedding_and_label = pd.concat([pd.DataFrame(x_embedding), pd.DataFrame(data=label,columns=["label"])], axis=1)
        sns.FacetGrid(embedding_and_label, hue="label", height=6).map(plt.scatter, 0, 1).add_legend()
        plt.savefig("{}.{}".format(key,extension))
        plt.clf()
    plt.close('all')

def labels2ply(filename, xyz, labels, seed=0, color_map:list=None):
    """
    Parameters
    ----------

    """
    np.random.seed(seed)
    max_labels = np.max(labels)
    if color_map is None:
        label_color_list = np.random.randint(0,255,(max_labels+1,3))
    else:
        label_color_list = color_map
    point_colors = np.array([ np.array([0,0,0]) if 0 > i else label_color_list[i] for i in labels])
    write_ply(filename, xyz, point_colors)

def intensity2ply(file_name, xyz, intensity):
    intensity = intensity[:,np.newaxis]
    intensity = np.concatenate([intensity,intensity,intensity],axis=1)
    rgb = (intensity*255)
    rgb = rgb.astype(np.int32)
    write_ply(file_name,xyz,rgb)

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

def dict2tensorboard(log_dict, writer, step):
    for key in log_dict:
        if isinstance(log_dict[key], list):
            _log = log_dict[key][0]
            _step = log_dict[key][1]
        else:
            _log = log_dict[key]
            _step = step
        writer.add_scalar(key, _log, _step)

def dict2wandb(log_dict, writer, step):
    writer.log(log_dict, step=step)

def dict2logger(log_dict, writer, step, log_service):
    # assert log_service in ["wandb","tensorboardX"]
    if log_service == "wandb":
        dict2wandb(log_dict, writer, step)
    elif log_service == "tensorboardX":
        dict2tensorboard(log_dict, writer, step)
    else:
        raise NotImplementedError("Unknown log_service: {}".format(log_service))

def dict2mdtables(tables, file_name, open_mode="w"):
    """
    Examples
    --------
    {
        "Result 1":{
            "head":["PC","Mac","Windows","Ubuntu"],
            "body":[
                ["X1","x","o","o"],
                ["Surface","x","o","o"],
                ["MacBook Air","o","o","o"]
            ],
        },
        "Result 2":{
            ...
        }
    }
    """
    with open(file_name, mode=open_mode) as f:
        for key in tables:
            f.write("\n{}\n\n".format(key))
            create_markdown_table(f, tables[key])
        f.close()

def dict2mdtable(table, file_name, open_mode="w"):
    """
    Examples
    --------
    {
        "head":["PC","Mac","Windows","Ubuntu"],
        "body":[
            ["X1","x","o","o"],
            ["Surface","x","o","o"],
            ["MacBook Air","o","o","o"]
        ]
    }
    """
    with open(file_name, mode=open_mode) as f:
       create_markdown_table(f, table)
       f.close()

def create_markdown_table(f, table):
    """
    Examples
    --------
    {
        "head":["PC","Mac","Windows","Ubuntu"],
        "body":[
            ["X1","x","o","o"],
            ["Surface","x","o","o"],
            ["MacBook Air","o","o","o"]
        ]
    }
    """
    head = table["head"]
    f.write("|"+"|".join([str(n) for n in head])+"|\n")
    f.write("|"+"".join(["-|" for i in range(len(head))])+"\n")
    body = table["body"]
    for row in body:
        f.write("|"+"|".join([str(n) for n in row])+"|\n")




