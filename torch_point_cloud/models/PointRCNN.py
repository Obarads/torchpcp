import torch
from torch import nn

from torch_point_cloud.modules import RPN
# from torch_point_cloud.modules import RCNNNet
from torch_point_cloud.configs.PointRCNN.config import cfg

class PointRCNN(nn.Module):
    def __init__(self, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)
        else:
            raise NotImplementedError

        return output


