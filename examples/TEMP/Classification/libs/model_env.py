import omegaconf

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# dataset
from torch_point_cloud.datasets.PointNet.ModelNet import (
    ModelNet40, 
    rotation_and_jitter
)

# model
from torch_point_cloud.models.PointNet import PointNetClassification

# loss function
from torch_point_cloud.losses.feature_transform_regularizer import feature_transform_regularizer

def get_dataset(cfg):
    """
    Get dataset.
    """
    if cfg.dataset.name == "modelnet40":
        dataset = ModelNet40(
            cfg.dataset.root,
            cfg.dataset.num_points,
        )
    else:
        raise NotImplementedError('Unknown cfg.dataset.name : ' + cfg.dataset.name)
    return dataset

def get_loader(cfg, dataset, with_collate_fn=False):
    """
    Split dataset into the batch size with DataLoader.
    """
    if with_collate_fn:
        collate_fn = rotation_and_jitter
    else:
        collate_fn = None

    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        shuffle=cfg.loader.shuffle,
        collate_fn=collate_fn,
    )

def get_model(cfg):
    """
    Get network model.
    """
    model = PointNetClassification(
        cfg.dataset.num_classes, 
        cfg.model.use_input_transform, 
        cfg.model.use_feature_transform
    )
    model.to(cfg.general.device)
    return model

def get_optimizer(cfg, model):
    """
    Get optimizer for training.
    """
    optimizer = optim.Adam(
        model.parameters(), 
        lr=cfg.optimizer.lr,
        weight_decay=1e-4
    )
    return optimizer

def get_scheduler(cfg, optimizer):
    """
    Get scheduler for training.
    """
    scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.scheduler.epoch_size, 
        gamma=cfg.scheduler.decay_rate
    )
    return scheduler

def get_losses(cfg):
    """
    Get losses.
    """
    # get losses
    criterion = Criterion(cfg)
    return criterion

class Criterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_feature_transform = cfg.model.use_feature_transform
        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_transform_reguliarzer = feature_transform_regularizer

    def forward(self, pred_cls_labels, feat_trans, cls_labels):
        loss = 0
        loss += self.cross_entropy(pred_cls_labels, cls_labels)
        if self.use_feature_transform:
            loss += self.feature_transform_reguliarzer(feat_trans) * 0.001
        return loss

def get_writer(cfg):
    """
    Get logger.
    """
    if cfg.writer.name == "tensorboardX":
        from tensorboardX import SummaryWriter
        writer = SummaryWriter("./")
    elif cfg.writer.name == "wandb":
        import wandb
        writer = wandb
        writer.init()
    else:
        raise NotImplementedError("Unknown cfg.writer.name : ", cfg.writer.name)
    return writer

def processing(model, criterion, data, meters, device, return_outputs=False):
    acc_meter, batch_loss = meters
    point_clouds, cls_labels = data

    # preprocessing of data
    point_clouds = torch.transpose(point_clouds, 1, 2)
    point_clouds = point_clouds.to(device, dtype=torch.float32)
    cls_labels = cls_labels.to(device, dtype=torch.long)

    # model forward processing
    pred_cls_labels, coord_trans, feat_trans = model(point_clouds)

    # compute losses with criterion
    loss = 0
    loss = criterion(pred_cls_labels, feat_trans, cls_labels)

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

def get_checkpoint(cfg):
    path = cfg.model.resume
    print("-> loading trained data '{}'".format(path))
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint_cfg = omegaconf.OmegaConf.create(checkpoint["cfg"])

    # set path and device
    checkpoint_cfg.dataset.root = cfg.dataset.root
    checkpoint_cfg.general.device = cfg.general.device

    return checkpoint, checkpoint_cfg

