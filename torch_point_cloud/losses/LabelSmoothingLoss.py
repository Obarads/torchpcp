import torch
from torch import nn
from torch.nn import functional as F

# https://github.com/WangYueFt/dgcnn/blob/e96a7e26555c3212dbc5df8d8875f07228e1ccc2/pytorch/util.py#L16
class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps=0.2, reduction="mean"):
        super().__init__()
        assert reduction in ["sum", "mean", "none"]
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, gt):
        num_classes = pred.size(1)
        eps = self.eps
        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        
        return loss
