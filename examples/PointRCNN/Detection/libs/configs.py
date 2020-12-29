##
## ref: https://hydra.cc/docs/next/tutorials/structured_config/config_groups
##

from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

##
## General
##

@dataclass
class General:
    ## epochs
    epochs: int = 200
    ## start epoch
    start_epoch: int = 0
    ## test and save term (If XX_epoch is -1, XX does not work.)
    save_epoch: int = 1
    ## seed value
    seed: int = 71
    ## reproducibility
    reproducibility: bool = False
    ## use cuda
    device: Any = "cuda"

##
## Model
##

@dataclass
class RPN:
    # from torch_point_cloud.models.PointNet import PointNetClassification for ModelNet40
    resume: str = MISSING

##
## Dataset
##

@dataclass
class KITTI:
    # dataset name
    name: str = "KITTI"
    # dataset path
    dataset_path: str = MISSING
    # gt_database path
    gt_database_path: str = MISSING
    # rpn num_points
    rpn_num_points: int = 16384
    rcnn_training_roi_dir: Any = None
    rcnn_training_feature_dir: Any = None
    

##
## Dataset Loader
##

@dataclass
class Loader:
    # loader cpu
    num_workers: int = 8
    # batch size
    batch_size: int = 16
    # data shuffle
    shuffle: bool = True

##
## Optimizer
##

@dataclass
class Optimizer:
    # learning rate
    lr: float = 0.001

##
## Scheduler
##

@dataclass
class Scheduler:
    # use torch.optim.lr_scheduler.StepLR
    # scheduler step_size
    epoch_size: int = 20
    # scheduler gamma
    decay_rate: float = 0.5

##
## Loss function
##

@dataclass
class Criterion:
    pass

##
## Logger
##

@dataclass
class Writer:
    name: str = "tensorboardX"

##
## Training Config
##

@dataclass
class KITTIConfig:
    general: Any = General
    model: Any = RPN
    dataset: Any = KITTI
    loader: Any = Loader
    optimizer: Any = Optimizer
    scheduler: Any = Scheduler
    criterion: Any = Criterion
    writer: Any = Writer

cs = ConfigStore.instance()
cs.store(name="kitti_config", node=KITTIConfig)

@hydra.main(config_name="kitti_config")
def check_config(cfg: KITTIConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    check_config()

