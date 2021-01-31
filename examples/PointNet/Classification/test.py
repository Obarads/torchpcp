import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "configs/test.yaml"))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import omegaconf
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# dataset
from torchpcp.datasets.PointNet.ModelNet import rotation_and_jitter

# tools
from torchpcp.utils.setting import (PytorchTools, get_configs,
                                             make_folders)
from torchpcp.utils.metrics import MultiAssessmentMeter, LossMeter

# env
from model_env import processing
from model_env import get_model, get_dataset, get_losses, get_checkpoint

def main():
    # get configs
    cfg, _, _ = get_configs(CONFIG_PATH)

    # make a output folder
    make_folders(cfg.output_folder)

    # set a seed 
    PytorchTools.set_seed(cfg.seed, cfg.device, cfg.reproducibility)

    # set a device
    cfg.device = PytorchTools.select_device(cfg.device)

    # get a trained model
    checkpoint, checkpoint_cfg = get_checkpoint(cfg.resume)

    model = get_model(checkpoint_cfg)
    dataset = get_dataset(checkpoint_cfg)
    criterion = get_losses()

    # set trained params
    model.load_state_dict(checkpoint["model"])

    eval(checkpoint_cfg, model, dataset["test"], criterion)
    print("Finish test.")

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

