import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs/train.yaml")
sys.path.append(os.path.join(BASE_DIR, "../../../")) # for package path

import omegaconf
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader

# dataset
from torch_point_cloud.datasets.PointNet.ModelNet import rotation_and_jitter

# tools
from torch_point_cloud.utils.setting import (PytorchTools, get_configs,
                                             make_folders)
from torch_point_cloud.utils.metrics import MultiAssessmentMeter, LossMeter
from torch_point_cloud.utils.converter import dict2tensorboard

# env
from model_env import processing, save_params
from model_env import get_model, get_dataset, get_losses, get_optimizer, get_scheduler

def main():
    # get configs
    cfg, _, _ = get_configs(CONFIG_PATH)

    # make a output folder
    make_folders(cfg.output_folder)

    # set a seed 
    PytorchTools.set_seed(cfg.seed, cfg.device, cfg.reproducibility)

    # set a device
    cfg.device = PytorchTools.select_device(cfg.device)

    model = get_model(cfg)
    dataset = get_dataset(cfg)
    criterion = get_losses()
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    # get a logger
    writer = SummaryWriter(cfg.output_folder)

    # start training
    for epoch in range(cfg.start_epoch, cfg.epochs):
        print('Epoch {}/{}:'.format(epoch, cfg.epochs))

        # training
        train_log = train(cfg, model, dataset["train"], optimizer, criterion, 
                          scheduler)

        dict2tensorboard(train_log, writer, epoch)

        # save params and model
        if (epoch+1) % cfg.save_epoch == 0 and cfg.save_epoch != -1:
            save_params(os.path.join(cfg.output_folder, "model.path.tar"), 
                        epoch+1, cfg, model, optimizer, scheduler)

    print('Epoch {}/{}:'.format(cfg.epochs, cfg.epochs))
    save_params(os.path.join(cfg.output_folder, "f_model.path.tar"), 
                cfg.epochs, cfg, model, optimizer, scheduler)
    writer.close()

    print("Finish training.")

# training
def train(cfg, model, dataset, optimizer, criterion, scheduler, publisher="train"):
    model.train()
    loader = DataLoader(
        #Subset(dataset["train"],range(320)),
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.nworkers,
        pin_memory=True,
        shuffle=True,
        collate_fn=rotation_and_jitter
    )
    loader = tqdm(loader, ncols=100, desc=publisher)

    acc_meter = MultiAssessmentMeter(num_classes=cfg.num_classes, 
                                     metrics=["class","overall"])
    batch_loss = LossMeter()
    meters = (acc_meter, batch_loss)

    for lidx, (point_clouds, labels) in enumerate(loader):
        optimizer.zero_grad()

        data = (point_clouds, labels)
        loss = processing(cfg, model, criterion, data, meters)

        loss.backward()
        optimizer.step()

        # nan
        if torch.isnan(loss):
            print("Train loss is nan.")
            exit()

    # update lr step
    scheduler.step()

    # get epoch loss and acc
    train_loss = batch_loss.compute()
    train_acc = acc_meter.compute()
    print('-> Train loss: {}  mAcc: {} oAcc: {}'.format(train_loss, 
                                                        train_acc["class"],
                                                        train_acc["overall"]))

    # save loss and acc to tensorboard
    lr = scheduler.get_last_lr()[0]
    log_dict = {
        "lr": lr,
        "train/loss": train_loss,
        "train/mAcc": train_acc["class"],
        "train/oAcc": train_acc["overall"]
    }

    return log_dict


if __name__ == "__main__":
    main()

