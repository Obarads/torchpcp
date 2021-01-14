import torch
import numpy as np

from libs import tpcpath
from torch_point_cloud.utils import pytorch_tools
# pytorch_tools.set_seed(0)
device = pytorch_tools.select_device("cuda")

from libs.dataset import SimpleSceneDataset
from torch.utils.data import DataLoader
from torch_point_cloud.modules.functional import furthest_point_sample, index2points
from torch_point_cloud.modules.functional import py_k_nearest_neighbors
from torch_point_cloud.modules.functional import k_nearest_neighbors as knn
from torch_point_cloud.utils.monitor import timecheck
from torch_point_cloud.modules.functional.sampling import gather

# PointRCNN impl. from https://github.com/sshaoshuai/PointRCNN
import pointnet2_cuda as pointnet2
from typing import Tuple

dataset = SimpleSceneDataset()
loader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=8,
    pin_memory=True,
    shuffle=False
)

for data in loader:
    point_clouds, sem_label, ins_label = data
    point_clouds = point_clouds[:, :3].to(device)
    center_idxs = furthest_point_sample(point_clouds, 1024)
    outs = gather(point_clouds, center_idxs)
    print(outs)
    exit()


