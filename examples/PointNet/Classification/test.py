import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../", "configs/modelnet40.yaml")
sys.path.append(os.path.join(BASE_DIR, "../")) # for package path
sys.path.append(os.path.join(BASE_DIR, "../../../")) # for package path

import omegaconf
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn

# dataset
from torch_point_cloud.datasets.PointNet.ModelNet import (
    ModelNet40, rotation_and_jitter)

# model
from torch_point_cloud.models.PointNet import PointNetClassification

# loss
from torch_point_cloud.losses.feature_transform_regularizer import (
    feature_transform_regularizer)

# tools
from torch_point_cloud.utils.setting import (PytorchTools, get_configs,
                                             make_folders)
from torch_point_cloud.utils.metrics import MultiAssessmentMeter, LossMeter
from torch_point_cloud.utils.converter import dict2tensorboard

# env
from model_env import processing, save_params


def main():
    # get configs
    cfg, _, _ = get_configs(CONFIG_PATH)

    # make a output folder
    make_folders(cfg.output_folder)

    # set a seed 
    PytorchTools.set_seed(cfg.seed, cfg.cuda, cfg.reproducibility)

    # set a device
    cfg.device = PytorchTools.select_device(cfg.device)

    # get a dataset
    dataset = ModelNet40(cfg.dataset.root, cfg.num_points)

    # get a model
    model = PointNetClassification(cfg.num_classes, cfg.num_points, 
                                   cfg.use_input_transform,
                                   cfg.use_feature_transform)

    # get losses
    criterion = {}
    criterion["cross_entropy"] = nn.CrossEntropyLoss()
    criterion["feature_transform_reguliarzer"] = feature_transform_regularizer

    eval(cfg, model, dataset["test"], criterion)
    print("Finish evaluation.")

# evaluation
def eval(cfg, model, dataset, criterion, publisher="test"):
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True
    )
    loader = tqdm(loader, ncols=100, desc=publisher)
    acc_meter = MultiAssessmentMeter(num_classes=cfg.num_classes, 
                                     metrics=["class","overall"])
    batch_loss = LossMeter()

    with torch.no_grad():
        for lidx, (point_clouds, labels) in enumerate(loader):
            data = (point_clouds, labels)
            meters = (acc_meter, batch_loss)
            _ = processing(cfg, model, criterion, data, meters)

    # get epoch loss and acc
    test_loss = batch_loss.compute()
    test_acc = acc_meter.compute()
    print('-> {} loss: {}  mAcc: {} oAcc: {}'.format(publisher, 
                                                     test_loss, 
                                                     test_acc["class"],
                                                     test_acc["overall"]))

if __name__ == "__main__":
    main()

