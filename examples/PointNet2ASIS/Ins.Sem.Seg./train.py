import os, sys
CW_DIR = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "configs/train.yaml"))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "../../../"))) # for package path

import hydra
import omegaconf
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader

# tools
from torchpcp.utils import pytorch_tools
from torchpcp.utils.monitor import dict2tensorboard, fix_path_in_configs
from torchpcp.utils.metrics import MultiAssessmentMeter, LossMeter

# env
from model_env import processing, save_params
from model_env import get_model, get_dataset, get_losses, get_optimizer, get_scheduler

@hydra.main(config_name=CONFIG_PATH)
def main(cfg:omegaconf.DictConfig):
    # fix paths
    cfg = fix_path_in_configs(CW_DIR, cfg, [["dataset","root"]])

    # set a seed 
    pytorch_tools.set_seed(
        cfg.general.seed, 
        cfg.general.device, 
        cfg.general.reproducibility
    )

    # set a device
    cfg.general.device = pytorch_tools.select_device(cfg.general.device)

    model = get_model(cfg)
    dataset = get_dataset(cfg)
    criterion = get_losses(cfg)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)

    # get a logger
    writer = SummaryWriter("./")

    # start training
    loader = tqdm(range(cfg.general.start_epoch, cfg.general.epochs), 
                  desc="Training", ncols=70)
    for epoch in loader:
        # print('Epoch {}/{}:'.format(epoch, cfg.general.epochs))

        # training
        train_log = train(
            cfg, 
            model, 
            dataset["train"], 
            optimizer, 
            criterion, 
            scheduler
        )

        dict2tensorboard(train_log, writer, epoch)

        # save params and model
        if (epoch+1) % cfg.general.save_epoch == 0 and \
            cfg.general.save_epoch != -1:
            save_params("model.path.tar", epoch+1, cfg, model, optimizer, 
                        scheduler)

    print('Epoch {}/{}:'.format(cfg.general.epochs, cfg.general.epochs))
    save_params("f_model.path.tar", cfg.general.epochs, cfg, model, optimizer, 
                scheduler)

    writer.close()

    print("Finish training.")

# training
def train(cfg, model, dataset, optimizer, criterion, scheduler, publisher="train"):
    model.train()
    loader = DataLoader(
        #Subset(dataset["train"],range(320)),
        dataset,
        batch_size=cfg.general.batch_size,
        num_workers=cfg.loader.nworkers,
        pin_memory=True,
        shuffle=True,
    )
    # loader = tqdm(loader, ncols=100, desc=publisher)

    acc_meter = MultiAssessmentMeter(
        num_classes=dataset.num_classes, 
        metrics=["class","overall","iou"]
    )
    batch_loss = LossMeter()
    meters = (acc_meter, batch_loss)
    print(len(loader))

    for data in loader:
        optimizer.zero_grad()

        loss = processing(model, criterion, data, meters, cfg.general.device)

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
    # print(
    #     '-> Train loss: {}  mAcc: {} iou: {} oAcc: {}'.format(
    #         train_loss, 
    #         train_acc["class"],
    #         train_acc["iou"],
    #         train_acc["overall"]
    #     )
    # )

    # save loss and acc to tensorboard
    lr = scheduler.get_last_lr()[0]
    log_dict = {
        "lr": lr,
        "train/loss": train_loss,
        "train/mAcc": train_acc["class"],
        "train/oAcc": train_acc["overall"],
        "train/IoU": train_acc["iou"]
    }

    return log_dict


if __name__ == "__main__":
    main()

