import omegaconf
import logging

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# dataset
from torch_point_cloud.datasets.PointRCNN.KITTI import KittiRCNNDataset

# model
from torch_point_cloud.models.PointRCNN import PointRCNN

# configs
from torch_point_cloud.configs.PointRCNN.config import cfg as ocfg

from libs.losses import Criterion

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def get_dataset(cfg, logger):
    """
    Get dataset.
    """
    if cfg.dataset.name == "KITTI":
        dataset = KittiRCNNDataset(
            root_dir=cfg.dataset.dataset_path, 
            npoints=ocfg.RPN.NUM_POINTS, 
            split=ocfg.TRAIN.SPLIT, 
            mode='TRAIN',
            logger=logger,
            classes=ocfg.CLASSES,
            rcnn_training_roi_dir=cfg.dataset.rcnn_training_roi_dir,
            rcnn_training_feature_dir=cfg.dataset.rcnn_training_feature_dir,
            gt_database_dir=cfg.dataset.gt_database_path
        )
    else:
        raise NotImplementedError('Unknown cfg.dataset.name : ' + cfg.dataset.name)
    return dataset

def get_loader(cfg, dataset):
    """
    Split dataset into the batch size with DataLoader.
    """
    collate_fn = dataset.collate_batch

    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=True,
        shuffle=cfg.loader.shuffle,
        collate_fn=collate_fn,
        drop_last=True
    )

def get_model(cfg):
    """
    Get network model.
    """
    model = PointRCNN()
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
    criterion = Criterion()
    return criterion

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
    sample_id = data["sample_id"]
    random_select = data["random_select"]
    aug_method = data["aug_method"]
    pts_input = data["pts_input"]
    pts_rect = data["pts_rect"]
    pts_features = data["pts_features"]
    rpn_cls_label = data["rpn_cls_label"]
    rpn_reg_label = data["rpn_reg_label"]
    gt_boxes3d = data["gt_boxes3d"]

    # numpy to torch tensor
    pts_input = torch.tensor(pts_input, dtype=torch.float32)
    gt_boxes3d = torch.tensor(gt_boxes3d, dtype=torch.float32)
    rpn_cls_label = torch.tensor(rpn_cls_label, dtype=torch.float32)
    rpn_reg_label = torch.tensor(rpn_reg_label, dtype=torch.float32)
    # change device
    pts_input = pts_input.to(device)
    gt_boxes3d = gt_boxes3d.to(device)
    rpn_cls_label = rpn_cls_label.to(device)
    rpn_reg_label = rpn_reg_label.to(device)
    # permute pts_input
    pts_input = pts_input.permute(0, 2, 1).contiguous()
    input_data = {'pts_input': pts_input, 'gt_boxes3d': gt_boxes3d}

    ret_dict = model(input_data)

    # compute losses with criterion
    rpn_cls = ret_dict["rpn_cls"]
    rpn_reg = ret_dict["rpn_reg"]

    tb_dict = {}
    loss = criterion(rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict)

    # save metrics
    if meters is not None:
        acc_meter, batch_loss = meters
        batch_loss.update(loss.item())
        # acc_meter.update(pred_cls_labels, cls_labels)

    if return_outputs:
        return loss, rpn_reg, rpn_reg
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

