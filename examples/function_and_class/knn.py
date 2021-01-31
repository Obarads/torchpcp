import numpy as np

import torch
from torch.utils.data import DataLoader

# local package
from libs import tpcpath
from libs.dataset import SimpleSceneDataset
from libs.three_nn import three_nn # PointRCNN

# torch-points-kernels
import torch_points_kernels as tpk

# torchpcp pacakage
from torchpcp.modules.functional import furthest_point_sample, index2points
from torchpcp.modules.functional import py_k_nearest_neighbors
from torchpcp.modules.functional.nns import k_nearest_neighbors as knn
from torchpcp.utils.monitor import timecheck
from torchpcp.utils import pytorch_tools

# pytorch_tools.set_seed(0)
device = pytorch_tools.select_device("cuda")

def speed_test(method, loader):
    for i, data in enumerate(loader): pass # for speed processing
    
    # print name
    if method == 0:
        t_name = "original c++ impl. time"
    elif method == 1:
        t_name = "original py impl. time"
    elif method == 2:
        t_name = "other c++ impl. time"
    elif method == 3:
        t_name = "tpk impl. time"
    else:
        raise NotImplementedError()

    # timer start
    t = timecheck()
    for _ in range(100):
        for i, data in enumerate(loader):
            point_clouds, sem_labels, ins_labels = data
            point_clouds = point_clouds[:, :3].to(device)
            center_idxs = furthest_point_sample(point_clouds, 1024)
            center_pc = index2points(point_clouds, center_idxs)
            if method == 0:
                _ = knn(center_pc, point_clouds, k=3)
            elif method == 1:
                _ = py_k_nearest_neighbors(center_pc, point_clouds, k=3, memory_saving=False)
            elif method == 2:
                _ = three_nn(center_pc.transpose(1,2).contiguous(), point_clouds.transpose(1,2).contiguous())
            elif method == 3:
                _ = tpk.knn(point_clouds.transpose(1,2).contiguous(), center_pc.transpose(1,2).contiguous(), 3)
            else:
                raise NotImplementedError()
    # timer end
    timecheck(t, t_name)

dataset = SimpleSceneDataset()
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    shuffle=False
)

import torch
from torch_geometric.nn import knn

x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
batch_x = torch.tensor([0, 0, 0, 0])
y = torch.Tensor([[-1, 0], [1, 0]])
batch_y = torch.tensor([0, 0])
assign_index = knn(x, y, 2, batch_x, batch_y)

print(assign_index)

# speed_test(0, loader)
# speed_test(1, loader)
# speed_test(2, loader)
# speed_test(3, loader)
