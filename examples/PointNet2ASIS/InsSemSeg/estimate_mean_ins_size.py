# To estimate the mean instance size of each class in training set
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "configs/estimate_mean_ins_size.yaml"))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import hydra
import omegaconf
import numpy as np
from scipy import stats

# dataset tools
from torchpcp.datasets.ASIS.utils import provider
from torchpcp.datasets.ASIS.S3DIS import get_block_paths

# tools
from torchpcp.utils import pytorch_tools as PytorchTools

@hydra.main(config_name=CONFIG_PATH)
def estimate(cfg:omegaconf.DictConfig):
    PytorchTools.set_seed(cfg.general.seed)

    h5_path_list = get_block_paths(cfg.dataset.root, cfg.dataset.s3dis.areas, ".h5")

    num_classes = cfg.dataset.s3dis.num_classes
    mean_ins_size = np.zeros(num_classes)
    ptsnum_in_gt = [[] for itmp in range(num_classes)]

    for h5_filename in h5_path_list:
        cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
        cur_data = np.reshape(cur_data, [-1, cur_data.shape[-1]])
        cur_group = np.reshape(cur_group, [-1])
        cur_sem = np.reshape(cur_sem, [-1])

        un = np.unique(cur_group)
        for ig, g in enumerate(un):
            tmp = (cur_group == g)
            sem_seg_g = int(stats.mode(cur_sem[tmp])[0])
            ptsnum_in_gt[sem_seg_g].append(np.sum(tmp))

    for idx in range(num_classes):
        mean_ins_size[idx] = np.mean(ptsnum_in_gt[idx]).astype(np.int)

    print(mean_ins_size)
    np.savetxt('mean_ins_size.txt', mean_ins_size)


if __name__ == "__main__":
    estimate()
