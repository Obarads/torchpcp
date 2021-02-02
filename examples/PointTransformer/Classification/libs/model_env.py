import omegaconf

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# dataset
from torchpcp.datasets.PointNet.ModelNet import ModelNet40 as PNModelNet40
from torchpcp.datasets.PointNet.ModelNet import rotation_and_jitter
from torchpcp.datasets.PointNet2.ModelNet import ModelNet40 as PN2ModelNet40
from torchpcp.datasets.PointNet2.ModelNet import MyModelNet40

# model
from torchpcp.models.PointTransformer import PointTransformerClassification
from torchpcp.utils.io.ply import write_pc
from torchpcp.utils import pytorch_tools

def get_dataset(cfg):
    """
    Get training and test dataset.
    """
    if cfg.dataset.name == "modelnet40":
        if cfg.dataset.mode == "pointnet2":
            dataset = MyModelNet40(cfg.dataset.root)
            train_collate_fn = dataset["train"].augment_batch_data
        elif cfg.dataset.mode == "pointnet":
            dataset = PNModelNet40(cfg.dataset.root, cfg.dataset.num_points)
            train_collate_fn = rotation_and_jitter
        else:
            raise NotImplementedError('Unknown cfg.dataset.mode : ' + cfg.dataset.mode)
    else:
        raise NotImplementedError('Unknown cfg.dataset.name : ' + cfg.dataset.name)
    return dataset, train_collate_fn

def get_loader(cfg, dataset, shuffle=False, collate_fn=None):
    """
    Split dataset into the batch size with DataLoader.
    """

    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )

def get_model(cfg):
    """
    Get network model.
    """
    model = PointTransformerClassification(
        in_channel_size=cfg.model.in_channel_size
    )
    model.to(cfg.general.device)
    return model

def get_optimizer(cfg, model):
    """
    Get optimizer for training.
    """
    # optimizer = optim.Adam(
    #     model.parameters(), 
    #     lr=cfg.optimizer.lr,
    #     weight_decay=1e-4
    # )
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=0.9,
        weight_decay=0.0001
    )
    return optimizer

def get_scheduler(cfg, optimizer):
    """
    Get scheduler for training.
    """
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, 
        # milestones=cfg.scheduler.epoch_list,
        milestones=[120,160],
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
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred_cls_labels, cls_labels):
        loss = 0
        loss += self.cross_entropy(pred_cls_labels, cls_labels)
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

    xyz = point_clouds[:, :3]
    # for i, xs in enumerate(xyz):
    #     p = pytorch_tools.t2n(xs.transpose(0,1))
    #     write_pc("{}.ply".format(i), p)

    # model forward processing
    pred_cls_labels = model(point_clouds, xyz)

    # compute losses with criterion
    loss = 0
    loss = criterion(pred_cls_labels, cls_labels)

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

