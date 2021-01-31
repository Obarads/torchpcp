import omegaconf

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler

# dataset
from torchpcp.datasets.PointNet import ModelNet

# model
from torchpcp.models.PointNet2 import PointNet2SSGClassification

# loss
from torchpcp.losses.LabelSmoothingLoss import LabelSmoothingLoss

def get_dataset(cfg):
    if cfg.dataset.name == "modelnet40":
        dataset = ModelNet.ModelNet40(
            cfg.dataset.root,
            cfg.dataset.num_points
        )
    else:
        raise NotImplementedError('Unknown dataset: ' + cfg.dataset.name)
    return dataset

def get_model(cfg):
    dataset_name = cfg.dataset.name
    num_classes = cfg.dataset[dataset_name].num_classes
    if cfg.model.name == "pointnet2ssg":
        model = PointNet2SSGClassification(
            num_classes, 
            cfg.model.point_feature_size
        )
    else:
        raise NotImplementedError('Unknown model: ' + cfg.model.name)
    model.to(cfg.general.device)
    return model

def get_optimizer(cfg, model):
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.optimizer.lr,
    )
    # optimizer = optim.SGD(
    #     model.parameters(), 
    #     lr=cfg.optimizer.lr*100, 
    #     momentum=cfg.optimizer.momentum, 
    #     weight_decay=1e-4
    # )

    return optimizer

def get_scheduler(cfg, optimizer):
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.scheduler.epoch_size, 
        gamma=cfg.scheduler.decay_rate
    )
    return scheduler

def get_losses(cfg):
    # get losses
    criterion = {}
    criterion["cross_entropy"] = nn.CrossEntropyLoss()
    return criterion

def processing(model, criterion, data, meters, device, return_outputs=False):
    acc_meter, batch_loss = meters
    point_clouds, cls_labels = data

    # preprocessing of data
    point_clouds = torch.transpose(point_clouds, 1, 2)
    point_clouds = point_clouds.to(device, dtype=torch.float32)
    cls_labels = cls_labels.to(device, dtype=torch.long)

    # model forward processing
    pred_cls_labels = model(point_clouds)

    # compute losses with criterion
    loss = 0
    loss += criterion["cross_entropy"](pred_cls_labels, cls_labels)
    # loss += criterion["label_smoothing"](pred_cls_labels, cls_labels)

    # save metrics
    batch_loss.update(loss.item())
    acc_meter.update(pred_cls_labels, cls_labels)

    if return_outputs:
        return loss, pred_cls_labels
    else:
        return loss

def save_params(model_path, epoch, cfg, model, optimizer, scheduler):
    torch.save({
        'epoch': epoch,
        'cfg': omegaconf.OmegaConf.to_container(cfg),
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, model_path)

def get_checkpoint(path):
    print("-> loading trained data '{}'".format(path))
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint_cfg = omegaconf.OmegaConf.create(checkpoint["cfg"])
    return checkpoint, checkpoint_cfg


