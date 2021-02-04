##
## ref: https://hydra.cc/docs/next/tutorials/structured_config/config_groups
##

from dataclasses import dataclass, field
from typing import Any, List

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
## Dataset
##

@dataclass
class ModelNet40:
    # dataset mode (pointnet, pointnet2)
    mode: str = "pointnet2"
    # dataset name
    name: str = "modelnet40"
    # number of classes
    num_classes: int = 40
    # dataset path
    root: str = MISSING
    # number of points
    num_points: int = 1024

##
## Model
##

# model parameters from dataset
npoint = ModelNet40.num_points
out_channels = ModelNet40.num_classes

# model parameters
NUM_POINTS = [npoint, npoint//4, npoint//16, npoint//64, npoint//256]
ENCODER_CHANNELS = [32, 64, 128, 256, 512] # output channel size
NUM_K_NEIGHBORS = [16, 16, 16, 16, 16]
DECODER_CHANNELS = [512, 256, out_channels] # decoder output channel size and dropout

# https://hydra.cc/docs/next/tutorials/structured_config/defaults/
@dataclass
class Model:
    # for test
    resume: str = MISSING
    # model parameters
    in_channel_size: int = 6
    in_num_point: int = npoint
    coord_channel_size: int = 3
    num_points: List[Any] = field(default_factory=lambda: NUM_POINTS)
    encoder_channel_sizes: List[Any] = field(default_factory=lambda: ENCODER_CHANNELS)
    bottleneck_ratio: int = 4
    num_k_neighbors: List[Any] = field(default_factory=lambda: NUM_K_NEIGHBORS)
    decoder_channel_sizes: List[Any] = field(default_factory=lambda: DECODER_CHANNELS)

##
## Dataset Loader
##

@dataclass
class Loader:
    # loader cpu
    num_workers: int = 8
    # batch size
    batch_size: int = 32
    # data shuffle
    shuffle: bool = True

##
## Optimizer
##

@dataclass
class Optimizer:
    # learning rate
    lr: float = 0.05
    # momentum
    momentum: float = 0.9
    # weight_decay
    weight_decay: float = 0.0001

##
## Scheduler
##

# shceduler param
EPOCH_LIST = [120, 160]

@dataclass
class Scheduler:
    # use torch.optim.lr_scheduler.StepLR
    # scheduler step_size
    epoch_list: List[Any] = field(default_factory=lambda: EPOCH_LIST)
    # scheduler gamma
    decay_rate: float = 0.1

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
class ModelNet40Config:
    general: Any = General
    model: Any = Model
    dataset: Any = ModelNet40
    loader: Any = Loader
    optimizer: Any = Optimizer
    scheduler: Any = Scheduler
    criterion: Any = Criterion
    writer: Any = Writer

cs = ConfigStore.instance()
cs.store(name="modelnet40_config", node=ModelNet40Config)

@hydra.main(config_name="modelnet40_config")
def check_config(cfg: ModelNet40Config) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    check_config()

