import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from lib.rpn.proposal_layer import ProposalLayer
# import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
# import lib.utils.loss_utils as loss_utils
# from lib.config import cfg
# import importlib

# https://github.com/sshaoshuai/PointRCNN/blob/master/lib/net/rpn.py

from torch_point_cloud.models.newPointNet2 import PointNet2MSGSemanticSegmentation
from torch_point_cloud.modules.Layer import PointwiseConv1D
# from torch_point_cloud.modules.ProposalLayer import RPNProposalLayer

from torch_point_cloud.configs.PointRCNN.config import cfg
from torch_point_cloud.utils.monitor import timecheck

class RPN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()
        self.training_mode = (mode == 'TRAIN')

        # MODEL = importlib.import_module(cfg.RPN.BACKBONE)
        # self.backbone_net = MODEL.get_model(input_channels=int(cfg.RPN.USE_INTENSITY), use_xyz=use_xyz)

        self.point_feature_size = int(cfg.RPN.USE_INTENSITY)
        self.backbone_net = PointNet2MSGSemanticSegmentation(self.point_feature_size)

        # classification branch
        # cls_layers = []
        # pre_channel = cfg.RPN.FP_MLPS[0][-1]
        # for k in range(0, cfg.RPN.CLS_FC.__len__()):
        #     cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.CLS_FC[k], bn=cfg.RPN.USE_BN))
        #     pre_channel = cfg.RPN.CLS_FC[k]
        # cls_layers.append(pt_utils.Conv1d(pre_channel, 1, activation=None))
        # if cfg.RPN.DP_RATIO >= 0:
        #     cls_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        # self.rpn_cls_layer = nn.Sequential(*cls_layers)

        cls_branch = [
            PointwiseConv1D(128, 128),
            nn.Dropout(0.5),
            PointwiseConv1D(128, 1, act=None),
        ]
        self.cls_branch = nn.Sequential(*cls_branch)

        # regression branch
        # per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        # if cfg.RPN.LOC_XZ_FINE:
        #     reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        # else:
        #     reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        # reg_channel += 1  # reg y

        # reg_layers = []
        # pre_channel = cfg.RPN.FP_MLPS[0][-1]
        # for k in range(0, cfg.RPN.REG_FC.__len__()):
        #     reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RPN.REG_FC[k], bn=cfg.RPN.USE_BN))
        #     pre_channel = cfg.RPN.REG_FC[k]
        # reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        # if cfg.RPN.DP_RATIO >= 0:
        #     reg_layers.insert(1, nn.Dropout(cfg.RPN.DP_RATIO))
        # self.rpn_reg_layer = nn.Sequential(*reg_layers)

        # regression branch
        per_loc_bin_num = int(cfg.RPN.LOC_SCOPE / cfg.RPN.LOC_BIN_SIZE) * 2
        if cfg.RPN.LOC_XZ_FINE:
            reg_channel = per_loc_bin_num * 4 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        else:
            reg_channel = per_loc_bin_num * 2 + cfg.RPN.NUM_HEAD_BIN * 2 + 3
        reg_channel += 1  # reg y

        reg_branch = [
            PointwiseConv1D(128, 128),
            nn.Dropout(0.5),
            PointwiseConv1D(128, reg_channel, act=None)
        ]
        self.reg_branch = nn.Sequential(*reg_branch)

        # if cfg.RPN.LOSS_CLS == 'DiceLoss':
        #     self.rpn_cls_loss_func = loss_utils.DiceLoss(ignore_target=-1)
        # elif cfg.RPN.LOSS_CLS == 'SigmoidFocalLoss':
        #     self.rpn_cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RPN.FOCAL_ALPHA[0],
        #                                                                        gamma=cfg.RPN.FOCAL_GAMMA)
        # elif cfg.RPN.LOSS_CLS == 'BinaryCrossEntropy':
        #     self.rpn_cls_loss_func = F.binary_cross_entropy
        # else:
        #     raise NotImplementedError

        # self.proposal_layer = RPNProposalLayer(mode=mode)
        self.init_weights()

    def init_weights(self):
        # if cfg.RPN.LOSS_CLS in ['SigmoidFocalLoss']:
        #     pi = 0.01
        #     nn.init.constant_(self.rpn_cls_layer[2].conv.bias, -np.log((1 - pi) / pi))

        # nn.init.normal_(self.rpn_reg_layer[-1].conv.weight, mean=0, std=0.001)
        pass

    def forward(self, input_data):
        """
        :param input_data: dict (point_cloud)
        :return:
        """
        pts_input = input_data['pts_input'] # xyz and intensity
        xyz = pts_input[:, 0:3]
        if self.point_feature_size > 0:
            features = pts_input[:, 3:]
        else:
            features = None

        backbone_features = self.backbone_net(xyz, features)
        backbone_xyz = None

        rpn_cls = self.cls_branch(backbone_features).transpose(1, 2).contiguous()  # (B, N, 1)
        rpn_reg = self.reg_branch(backbone_features).transpose(1, 2).contiguous()  # (B, N, C)

        ret_dict = {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                    'backbone_xyz': backbone_xyz, 'backbone_features': backbone_features}

        return ret_dict

