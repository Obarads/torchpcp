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
    epochs: int = 250
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
class Model:
    # from torchpcp.models.PointNet import PointNetClassification for ModelNet40
    output_dim: int = 40
    use_input_transform: bool = True
    use_feature_transform: bool = True
    resume: str = MISSING

##
## Dataset
##

@dataclass
class ModelNet40:
    # dataset name
    name: str = "modelnet40"
    # number of classes
    num_classes: int = 40
    # dataset path
    root: str = MISSING
    # number of points
    num_points: int = 1024

##
## Dataset Loader
##

@dataclass
class Loader:
    # loader cpu
    num_workers: int = 4
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

