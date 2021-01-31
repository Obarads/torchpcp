import os, sys
CW_DIR = os.getcwd()

from tqdm import tqdm
import hydra
import torch

# add path
import libs.tpcpath

# tools
from torchpcp.utils import monitor, pytorch_tools
from torchpcp.utils.metrics import MultiAssessmentMeter, LossMeter
from torchpcp.configs.PointRCNN.config import cfg as ocfg

# env
from libs.model_env import processing, save_params
from libs.model_env import (get_model, get_optimizer, get_scheduler, get_losses, 
                            get_dataset, get_writer, get_loader, create_logger)

# config
from libs import configs

@hydra.main(config_name="kitti_config")
def main(cfg: configs.KITTIConfig) -> None:
    # ifx paths
    cfg = monitor.fix_path_in_configs(CW_DIR, cfg, [["dataset", "dataset_path"]])

    # set a seed
    pytorch_tools.set_seed(
        cfg.general.seed,
        cfg.general.device,
        cfg.general.reproducibility
    )

    # set a device
    cfg.general.device = pytorch_tools.select_device(cfg.general.device)

    ## I will remove this code.
    log_file = os.path.join("./", 'log_train.txt')
    logger = create_logger(log_file)

    # training env
    ## Get model.
    model = get_model(cfg)
    ## Get dataset and loader.
    dataset = get_dataset(cfg, logger)
    datset_loader = get_loader(cfg, dataset)
    ## Get loss functions.
    criterion = get_losses(cfg)
    ## Get optimizer.
    optimizer = get_optimizer(cfg, model)
    ## Get scheduler.
    scheduler = get_scheduler(cfg, optimizer)
    ## Get logger.
    writer = get_writer(cfg)

    # progress bar
    loader = tqdm(range(cfg.general.start_epoch, cfg.general.epochs),
                  desc="Training", ncols=70)

    # training start
    for epoch in loader:
        # training
        train_log = train(cfg, model, datset_loader, optimizer, criterion,
                          scheduler)

        # save training log
        monitor.dict2logger(train_log, writer, epoch, cfg.writer.name)

        # save params and model
        if (epoch+1) % cfg.general.save_epoch == 0 and \
            cfg.general.save_epoch != -1:
            save_params("model.path.tar", epoch+1, cfg, model, optimizer, 
                        scheduler)

    print('Epoch {}/{}:'.format(cfg.general.epochs, cfg.general.epochs))
    save_params("f_model.path.tar", cfg.general.epochs, cfg, model, optimizer, 
                scheduler)

    print("Finish training.")


def train(cfg, model, loader, optimizer, criterion, scheduler, publisher="train"):
    model.train()

    # metrics
    acc_meter = MultiAssessmentMeter(
        # num_classes=cfg.dataset.num_classes, 
        num_classes=20, # Flaky
        metrics=["class","overall","iou"]
    )
    batch_loss = LossMeter()
    meters = (acc_meter, batch_loss)

    for _, data in enumerate(loader):
        optimizer.zero_grad()
        loss = processing(model, criterion, data, meters, cfg.general.device)
        loss.backward()
        optimizer.step()

        # nan
        if torch.isnan(loss):
            print("Training loss is nan.")
            exit()

    scheduler.step()

    # get epoch loss and accuracy
    epoch_loss = batch_loss.compute()
    # epoch_acc = acc_meter.compute()

    # save loss and acc to tensorboard
    lr = scheduler.get_last_lr()[0]
    log_dict = {
        "lr": lr,
        "{}/loss".format(publisher): epoch_loss,
    #     "{}/mAcc".format(publisher): epoch_acc["class"],
    #     "{}/oAcc".format(publisher): epoch_acc["overall"],
    #     "{}/IoU".format(publisher): epoch_acc["iou"]
    }

    return log_dict



if __name__ == "__main__":
    main()



