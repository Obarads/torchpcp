import os, sys
CW_DIR = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "configs/test.yaml"))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import hydra
import omegaconf
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader

# tools
from torchpcp.utils.setting import PytorchTools, fix_path_in_configs
from torchpcp.utils.metrics import MultiAssessmentMeter, LossMeter

# env
from model_env import processing
from model_env import get_model, get_dataset, get_losses, get_checkpoint

@hydra.main(config_name=CONFIG_PATH)
def main(cfg:omegaconf.DictConfig):
    # fix paths
    cfg = fix_path_in_configs(CW_DIR, cfg, 
                              [["dataset","root"],["model","resume"]])

    # set a seed 
    PytorchTools.set_seed(
        cfg.general.seed, 
        cfg.general.device, 
        cfg.general.reproducibility
    )

    # set a device
    cfg.general.device = PytorchTools.select_device(cfg.general.device)

    # get a trained model env
    checkpoint, checkpoint_cfg = get_checkpoint(cfg.model.resume)

    # change cfg
    checkpoint_cfg.dataset.root = cfg.dataset.root

    model = get_model(checkpoint_cfg)
    dataset = get_dataset(checkpoint_cfg)
    criterion = get_losses(checkpoint_cfg)

    # set trained params
    model.load_state_dict(checkpoint["model"])

    # start test
    test(cfg, model, dataset["test"], criterion)

    print("Finish test.")

# training
def test(cfg, model, dataset, criterion, publisher="test"):
    model.eval()
    loader = DataLoader(
        #Subset(dataset["train"],range(320)),
        dataset,
        batch_size=cfg.general.batch_size,
        num_workers=cfg.loader.nworkers,
        pin_memory=True,
    )
    # loader = tqdm(loader, ncols=100, desc=publisher)

    acc_meter = MultiAssessmentMeter(
        num_classes=dataset.num_classes, 
        metrics=["class","overall","iou"]
    )
    batch_loss = LossMeter()
    meters = (acc_meter, batch_loss)

    with torch.no_grad():
        for data in loader:
            loss = processing(model, criterion, data, meters, cfg.general.device)

    # get epoch loss and acc
    test_loss = batch_loss.compute()
    test_acc = acc_meter.compute()
    print(
        '-> {} loss: {}  mAcc: {} iou: {} oAcc: {}'.format(
            publisher,
            test_loss, 
            test_acc["class"],
            test_acc["iou"],
            test_acc["overall"]
        )
    )

    # save loss and acc to tensorboard
    log_dict = {
        "{}/loss".format(publisher):test_loss,
        "{}/mAcc".format(publisher):test_acc["class"],
        "{}/oAcc".format(publisher):test_acc["overall"],
        "{}/IoU".format(publisher):test_acc["iou"]
    }

    return log_dict


if __name__ == "__main__":
    main()

