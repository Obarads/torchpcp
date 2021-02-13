import torch
import numpy as np

from libs import tpcpath
from torchpcp.utils import pytorch_tools
# pytorch_tools.set_seed(0)
device = pytorch_tools.select_device("cuda")

from libs.dataset import SimpleSceneDataset
from torch.utils.data import DataLoader
from torchpcp.modules.functional.sampling import furthest_point_sampling
from torchpcp.modules.functional.nns import py_k_nearest_neighbors
from torchpcp.modules.functional.nns import k_nearest_neighbors
from torchpcp.utils.monitor import timecheck
from torchpcp.modules.functional import other
# from torchpcp.modules.functional.other import gather, index2points

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

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

# for data in loader:
#     point_clouds, sem_label, ins_label = data
#     point_clouds = point_clouds[:, :, :3].transpose(1,2).to(device)
#     center_idxs = furthest_point_sampling(point_clouds, 1024)
#     t = timecheck()
#     outs = gather(point_clouds, center_idxs)
#     t = timecheck(t, "gather:")
#     gt_outs = other.index2points(point_clouds, center_idxs)
#     t = timecheck(t, "index2points:")
#     acc = outs == gt_outs
#     print(False in (acc))
#     # print(acc)
#     # print(torch.sum(acc))
#     # print(outs.shape)
#     # print(outs)
#     exit()

k = 1
for data in loader:
    point_clouds, sem_label, ins_label = data
    point_clouds = point_clouds[:, :, :3].transpose(1,2).to(device)
    center_idxs = furthest_point_sampling(point_clouds, 1024)
    center_points = other.index2points(point_clouds, center_idxs)
    print(point_clouds.shape, center_points.shape)
    knn_idxs, _ = k_nearest_neighbors(center_points, point_clouds, k)
    t = timecheck()
    outs = other.gather(point_clouds, knn_idxs)
    t = timecheck(t, "gather:")
    gt_outs = other.index2points(point_clouds, knn_idxs)
    t = timecheck(t, "index2points:")
    acc = outs == gt_outs
    print(False in (acc))
    # print(acc)
    # print(torch.sum(acc))
    # print(outs.shape)
    # print(outs)
    exit()



