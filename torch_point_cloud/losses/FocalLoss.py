import torch 
from torch import nn
from torch.nn import functional as F

# https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

# https://github.com/richardaecn/class-balanced-loss/issues/2
def multi_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    # alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    inputs:[batch_size, num_classes, ...]
    targets:[batch_size, ...]
    """

    num_classes = inputs.shape[1]
    inputs = torch.transpose(inputs, 1, -1)
    targets = get_one_hot(targets, num_classes)
    targets = targets.to(dtype=torch.float32)

    p = torch.sigmoid(inputs)
    focal_weight = torch.pow((p-1)*targets + p*(1-targets), gamma)

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss * focal_weight

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def get_one_hot(labels, num_classes):
    one_hot = F.one_hot(labels.to(dtype=torch.int64), num_classes)
    return one_hot

class FocalLoss(nn.Module):
    def __init__(self, alpha=-1., gamma=2., reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = sigmoid_focal_loss(inputs, targets, alpha=self.alpha, 
                                  gamma=self.gamma, reduction=self.reduction)
        return loss

class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=2., reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = multi_sigmoid_focal_loss(inputs, targets, gamma=self.gamma, 
                                        reduction="none")
        B = inputs.shape[0]
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        return loss


