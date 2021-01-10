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

# PointRCNN impl. from https://github.com/sshaoshuai/PointRCNN
import pointnet2_cuda as pointnet2
from typing import Tuple
class ThreeNN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the three nearest neighbors of unknown in known
        :param ctx:
        :param unknown: (B, N, 3)
        :param known: (B, M, 3)
        :return:
            dist: (B, N, 3) l2 distance to the three nearest neighbors
            idx: (B, N, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

three_nn = ThreeNN.apply


def speed_test(method, loader):
    for i, data in enumerate(loader): pass # for speed processing
    
    # print name
    if method == 0:
        t_name = "original c++ impl. time"
    elif method == 1:
        t_name = "original py impl. time"
    elif method == 2:
        t_name = "other c++ impl. time"
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

speed_test(0, loader)
speed_test(1, loader)
speed_test(2, loader)
