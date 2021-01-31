import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

# configs
from torchpcp.configs.PointRCNN.config import cfg

# losses
from torchpcp.losses.FocalLoss import FocalLoss
from torchpcp.losses.DiceLoss import DiceLoss


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        if cfg.RPN.LOSS_CLS == "DiceLoss":
            self.rpn_cls_loss_func = DiceLoss(ignore_target=-1)
        elif cfg.RPN.LOSS_CLS == "SigmoidFocalLoss":
            self.rpn_cls_loss_func = FocalLoss()
        elif cfg.RPN.LOSS_CLS == "BinaryCrossEntropy":
            self.rpn_cls_loss_func = F.binary_cross_entropy
        else:
            raise NotImplementedError()

    def forward(self, rpn_cls, rpn_reg, rpn_cls_label, rpn_reg_label, tb_dict):
        # if isinstance(model, nn.DataParallel):
        #     rpn_cls_loss_func = model.module.rpn.rpn_cls_loss_func
        # else:
        #     rpn_cls_loss_func = model.rpn.rpn_cls_loss_func

        MEAN_SIZE = torch.from_numpy(cfg.CLS_MEAN_SIZE[0]).cuda()
        rpn_cls_label_flat = rpn_cls_label.view(-1)
        rpn_cls_flat = rpn_cls.view(-1)
        fg_mask = (rpn_cls_label_flat > 0)

        # RPN classification loss
        if cfg.RPN.LOSS_CLS == 'DiceLoss':
            rpn_loss_cls = self.rpn_cls_loss_func(rpn_cls, rpn_cls_label_flat)

        elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
            rpn_cls_target = (rpn_cls_label_flat > 0).float()
            pos = (rpn_cls_label_flat > 0).float()
            neg = (rpn_cls_label_flat == 0).float()
            cls_weights = pos + neg
            pos_normalizer = pos.sum()
            cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)
            rpn_loss_cls = self.rpn_cls_loss_func(
                rpn_cls_flat, rpn_cls_target, cls_weights)
            rpn_loss_cls_pos = (rpn_loss_cls * pos).sum()
            rpn_loss_cls_neg = (rpn_loss_cls * neg).sum()
            rpn_loss_cls = rpn_loss_cls.sum()
            tb_dict['rpn_loss_cls_pos'] = rpn_loss_cls_pos.item()
            tb_dict['rpn_loss_cls_neg'] = rpn_loss_cls_neg.item()

        elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
            weight = rpn_cls_flat.new(rpn_cls_flat.shape[0]).fill_(1.0)
            weight[fg_mask] = cfg.RPN.FG_WEIGHT
            rpn_cls_label_target = (rpn_cls_label_flat > 0).float()
            batch_loss_cls = self.rpn_cls_loss_func(torch.sigmoid(rpn_cls_flat), 
                                                    rpn_cls_label_target,
                                                    weight=weight, reduction='none')
            cls_valid_mask = (rpn_cls_label_flat >= 0).float()
            rpn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / \
                torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        # RPN regression loss
        point_num = rpn_reg.size(0) * rpn_reg.size(1)
        fg_sum = fg_mask.long().sum().item()
        if fg_sum != 0:
            loss_loc, loss_angle, loss_size, reg_loss_dict = \
                self.get_reg_loss(rpn_reg.view(point_num, -1)[fg_mask],
                                  rpn_reg_label.view(point_num, 7)[fg_mask],
                                  loc_scope=cfg.RPN.LOC_SCOPE,
                                  loc_bin_size=cfg.RPN.LOC_BIN_SIZE,
                                  num_head_bin=cfg.RPN.NUM_HEAD_BIN,
                                  anchor_size=MEAN_SIZE,
                                  get_xz_fine=cfg.RPN.LOC_XZ_FINE,
                                  get_y_by_bin=False,
                                  get_ry_fine=False)

            loss_size = 3 * loss_size  # consistent with old codes
            rpn_loss_reg = loss_loc + loss_angle + loss_size
        else:
            loss_loc = loss_angle = loss_size = rpn_loss_reg = rpn_loss_cls * 0

        rpn_loss = rpn_loss_cls * \
            cfg.RPN.LOSS_WEIGHT[0] + rpn_loss_reg * cfg.RPN.LOSS_WEIGHT[1]

        tb_dict.update({'rpn_loss_cls': rpn_loss_cls.item(), 'rpn_loss_reg': rpn_loss_reg.item(),
                        'rpn_loss': rpn_loss.item(), 'rpn_fg_sum': fg_sum, 'rpn_loss_loc': loss_loc.item(),
                        'rpn_loss_angle': loss_angle.item(), 'rpn_loss_size': loss_size.item()})

        return rpn_loss

    def get_reg_loss(self, pred_reg, reg_label, loc_scope, loc_bin_size,
                     num_head_bin, anchor_size, get_xz_fine=True,
                     get_y_by_bin=False, loc_y_scope=0.5,
                     loc_y_bin_size=0.25, get_ry_fine=False):
        """
        Bin-based 3D bounding boxes regression loss. See https://arxiv.org/abs/1812.04244 for more details.

        :param pred_reg: (N, C)
        :param reg_label: (N, 7) [dx, dy, dz, h, w, l, ry]
        :param loc_scope: constant
        :param loc_bin_size: constant
        :param num_head_bin: constant
        :param anchor_size: (N, 3) or (3)
        :param get_xz_fine:
        :param get_y_by_bin:
        :param loc_y_scope:
        :param loc_y_bin_size:
        :param get_ry_fine:
        :return:
        """
        per_loc_bin_num = int(loc_scope / loc_bin_size) * 2
        loc_y_bin_num = int(loc_y_scope / loc_y_bin_size) * 2

        reg_loss_dict = {}
        loc_loss = 0

        # xz localization loss
        x_offset_label, y_offset_label, z_offset_label = reg_label[:,
                                                                   0], reg_label[:, 1], reg_label[:, 2]
        x_shift = torch.clamp(x_offset_label + loc_scope,
                              0, loc_scope * 2 - 1e-3)
        z_shift = torch.clamp(z_offset_label + loc_scope,
                              0, loc_scope * 2 - 1e-3)
        x_bin_label = (x_shift / loc_bin_size).floor().long()
        z_bin_label = (z_shift / loc_bin_size).floor().long()

        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        loss_x_bin = F.cross_entropy(
            pred_reg[:, x_bin_l: x_bin_r], x_bin_label)
        loss_z_bin = F.cross_entropy(
            pred_reg[:, z_bin_l: z_bin_r], z_bin_label)
        reg_loss_dict['loss_x_bin'] = loss_x_bin.item()
        reg_loss_dict['loss_z_bin'] = loss_z_bin.item()
        loc_loss += loss_x_bin + loss_z_bin

        if get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r

            x_res_label = x_shift - \
                (x_bin_label.float() * loc_bin_size + loc_bin_size / 2)
            z_res_label = z_shift - \
                (z_bin_label.float() * loc_bin_size + loc_bin_size / 2)
            x_res_norm_label = x_res_label / loc_bin_size
            z_res_norm_label = z_res_label / loc_bin_size

            x_bin_onehot = torch.cuda.FloatTensor(
                x_bin_label.size(0), per_loc_bin_num).zero_()
            x_bin_onehot.scatter_(1, x_bin_label.view(-1, 1).long(), 1)
            z_bin_onehot = torch.cuda.FloatTensor(
                z_bin_label.size(0), per_loc_bin_num).zero_()
            z_bin_onehot.scatter_(1, z_bin_label.view(-1, 1).long(), 1)

            loss_x_res = F.smooth_l1_loss(
                (pred_reg[:, x_res_l: x_res_r] * x_bin_onehot).sum(dim=1), x_res_norm_label)
            loss_z_res = F.smooth_l1_loss(
                (pred_reg[:, z_res_l: z_res_r] * z_bin_onehot).sum(dim=1), z_res_norm_label)
            reg_loss_dict['loss_x_res'] = loss_x_res.item()
            reg_loss_dict['loss_z_res'] = loss_z_res.item()
            loc_loss += loss_x_res + loss_z_res

        # y localization loss
        if get_y_by_bin:
            y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
            y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
            start_offset = y_res_r

            y_shift = torch.clamp(
                y_offset_label + loc_y_scope, 0, loc_y_scope * 2 - 1e-3)
            y_bin_label = (y_shift / loc_y_bin_size).floor().long()
            y_res_label = y_shift - \
                (y_bin_label.float() * loc_y_bin_size + loc_y_bin_size / 2)
            y_res_norm_label = y_res_label / loc_y_bin_size

            y_bin_onehot = torch.cuda.FloatTensor(
                y_bin_label.size(0), loc_y_bin_num).zero_()
            y_bin_onehot.scatter_(1, y_bin_label.view(-1, 1).long(), 1)

            loss_y_bin = F.cross_entropy(
                pred_reg[:, y_bin_l: y_bin_r], y_bin_label)
            loss_y_res = F.smooth_l1_loss(
                (pred_reg[:, y_res_l: y_res_r] * y_bin_onehot).sum(dim=1), y_res_norm_label)

            reg_loss_dict['loss_y_bin'] = loss_y_bin.item()
            reg_loss_dict['loss_y_res'] = loss_y_res.item()

            loc_loss += loss_y_bin + loss_y_res
        else:
            y_offset_l, y_offset_r = start_offset, start_offset + 1
            start_offset = y_offset_r

            loss_y_offset = F.smooth_l1_loss(
                pred_reg[:, y_offset_l: y_offset_r].sum(dim=1), y_offset_label)
            reg_loss_dict['loss_y_offset'] = loss_y_offset.item()
            loc_loss += loss_y_offset

        # angle loss
        ry_bin_l, ry_bin_r = start_offset, start_offset + num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + num_head_bin

        ry_label = reg_label[:, 6]

        if get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / num_head_bin

            ry_label = ry_label % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (ry_label > np.pi * 0.5) & (ry_label < np.pi * 1.5)
            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            ry_label[opposite_flag] = (
                ry_label[opposite_flag] + np.pi) % (2 * np.pi)
            shift_angle = (ry_label + np.pi * 0.5) % (2 * np.pi)  # (0 ~ pi)

            shift_angle = torch.clamp(
                shift_angle - np.pi * 0.25, min=1e-3, max=np.pi * 0.5 - 1e-3)  # (0, pi/2)

            # bin center is (5, 10, 15, ..., 85)
            ry_bin_label = (shift_angle / angle_per_class).floor().long()
            ry_res_label = shift_angle - \
                (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
            ry_res_norm_label = ry_res_label / (angle_per_class / 2)

        else:
            # divide 2pi into several bins
            angle_per_class = (2 * np.pi) / num_head_bin
            heading_angle = ry_label % (2 * np.pi)  # 0 ~ 2pi

            shift_angle = (heading_angle + angle_per_class / 2) % (2 * np.pi)
            ry_bin_label = (shift_angle / angle_per_class).floor().long()
            ry_res_label = shift_angle - \
                (ry_bin_label.float() * angle_per_class + angle_per_class / 2)
            ry_res_norm_label = ry_res_label / (angle_per_class / 2)

        ry_bin_onehot = torch.cuda.FloatTensor(
            ry_bin_label.size(0), num_head_bin).zero_()
        ry_bin_onehot.scatter_(1, ry_bin_label.view(-1, 1).long(), 1)
        loss_ry_bin = F.cross_entropy(
            pred_reg[:, ry_bin_l:ry_bin_r], ry_bin_label)
        loss_ry_res = F.smooth_l1_loss(
            (pred_reg[:, ry_res_l: ry_res_r] * ry_bin_onehot).sum(dim=1), ry_res_norm_label)

        reg_loss_dict['loss_ry_bin'] = loss_ry_bin.item()
        reg_loss_dict['loss_ry_res'] = loss_ry_res.item()
        angle_loss = loss_ry_bin + loss_ry_res

        # size loss
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3
        assert pred_reg.shape[1] == size_res_r, '%d vs %d' % (
            pred_reg.shape[1], size_res_r)

        size_res_norm_label = (reg_label[:, 3:6] - anchor_size) / anchor_size
        size_res_norm = pred_reg[:, size_res_l:size_res_r]
        size_loss = F.smooth_l1_loss(size_res_norm, size_res_norm_label)

        # Total regression loss
        reg_loss_dict['loss_loc'] = loc_loss
        reg_loss_dict['loss_angle'] = angle_loss
        reg_loss_dict['loss_size'] = size_loss

        return loc_loss, angle_loss, size_loss, reg_loss_dict
