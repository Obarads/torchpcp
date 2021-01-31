import omegaconf

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler

# dataset
from torchpcp.datasets.PointNet.ModelNet import (
    ModelNet40, rotation_and_jitter)

# model
from torchpcp.models.PointNet import PointNetClassification

# loss
from torchpcp.losses.feature_transform_regularizer import (
    feature_transform_regularizer)

def get_dataset(cfg):
    if dataset.name == "ModelNet40":
        dataset = ModelNet40(cfg.dataset.root, cfg.num_points)
    else:
        raise NotImplementedError('Unknown dataset: ' + cfg.dataset.name)
    return dataset

def get_model(cfg):
    dataset_name = cfg.dataset.name
    num_classes = cfg.dataset[dataset_name].num_classes
    model = PointNetClassification(num_classes, cfg.num_points, 
                                   cfg.use_input_transform,
                                   cfg.use_feature_transform)
    model.to(cfg.device)
    return model

def get_optimizer(cfg, model):
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    return optimizer

def get_scheduler(cfg, optimizer):
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.epoch_size, 
                                    gamma=cfg.decay_rate)
    return scheduler

def get_losses():
    # get losses
    criterion = {}
    criterion["cross_entropy"] = nn.CrossEntropyLoss()
    criterion["feature_transform_reguliarzer"] = feature_transform_regularizer
    return criterion

def processing(cfg, model, criterion, data, meters):
    acc_meter, batch_loss = meters
    point_clouds, labels = data

    # preprocessing of data
    point_clouds = torch.transpose(point_clouds, 1, 2)
    point_clouds = point_clouds.to(cfg.device, dtype=torch.float32, non_blocking=True)
    labels = labels.to(cfg.device, dtype=torch.long, non_blocking=True)

    # model forward processing
    pred_labels, _, feat_trans = model(point_clouds)

    # compute losses with criterion
    loss = 0
    loss += criterion["cross_entropy"](pred_labels, labels)
    if cfg.use_feature_transform:
        loss += criterion["feature_transform_reguliarzer"](feat_trans) * 0.001

    # save metrics
    batch_loss.update(loss.item())
    acc_meter.update(pred_labels, labels)

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