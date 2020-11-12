import omegaconf

import torch
from torch import optim
from torch import nn
from torch.optim import lr_scheduler

# dataset
from torch_point_cloud.datasets.ASIS import S3DIS

# model
from torch_point_cloud.models.new_PointNet2ASIS import PointNet2ASIS

# loss
from torch_point_cloud.losses.DiscriminativeLoss import DiscriminativeLoss

def get_dataset(cfg):
    if cfg.dataset.name in ["s3dis", "s3dis_block"]:
        dataset = S3DIS.S3DISBlockDataset(
            cfg.dataset.root, 
            cfg.dataset.num_points,
            cfg.dataset.s3dis.areas,
            memory_saving=cfg.dataset.memory_saving
        )
    elif cfg.dataset.name in ["s3dis_scene"]:
        dataset = S3DIS.S3DISSceneDataset(
            cfg.dataset.root,
            cfg.dataset.num_points,
            cfg.dataset.s3dis.areas
        )
    else:
        raise NotImplementedError('Unknown dataset: ' + cfg.dataset.name)
    return dataset

def get_model(cfg):
    dataset_name = cfg.dataset.name.split("_")[0]
    num_classes = cfg.dataset[dataset_name].num_classes
    if cfg.model.name == "pointnet2asis":
        model = PointNet2ASIS(num_classes, cfg.model.ins_channel)
    else:
        raise NotImplementedError('Unknown model: ' + cfg.model.name)
    model.to(cfg.general.device)
    return model

def get_optimizer(cfg, model):
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
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
    criterion["discriminative"] = DiscriminativeLoss(
        cfg.criterion.discriminative.delta_d,
        cfg.criterion.discriminative.delta_v,
        alpha=cfg.criterion.discriminative.var,
        beta=cfg.criterion.discriminative.dist,
        gamma=cfg.criterion.discriminative.reg,
        norm_p=1
    )
    return criterion

def processing(cfg, model, criterion, data, meters):
    acc_meter, batch_loss = meters
    point_clouds, sem_labels, ins_labels, ins_label_sizes, ins_masks = data

    # if cfg.dataset.name == "S3DIS":
    #     point_clouds, sem_labels, ins_labels, ins_label_sizes, ins_masks = data
    # else:
    #     raise NotImplementedError('Unknown dataset: ' + cfg.dataset.name)

    # preprocessing of data
    point_clouds = torch.transpose(point_clouds, 1, 2)
    point_clouds = point_clouds.to(cfg.general.device, dtype=torch.float32)
    sem_labels = sem_labels.to(cfg.general.device, dtype=torch.long)
    ins_masks = ins_masks.to(cfg.general.device, dtype=torch.float32)

    # model forward processing
    pred_sem_labels, ins_embeds = model(point_clouds)

    # compute losses with criterion
    ins_embeds = ins_embeds.transpose(1,2).contiguous()
    loss = 0
    loss += criterion["cross_entropy"](pred_sem_labels, sem_labels)
    # print(ins_embeds.shape, ins_masks.shape, ins_label_sizes.shape)
    loss += criterion["discriminative"](ins_embeds, ins_masks, ins_label_sizes)

    # save metrics
    batch_loss.update(loss.item())
    acc_meter.update(pred_sem_labels, sem_labels)

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


