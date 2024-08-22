import torch
import torch.nn as nn
import torch.nn.functional as F

"""
NOTES:
Try and use loss functions which are less sensitive to outliers such as MAE or Huber loss over MSE. This will reduce the effect of mislabeled datapoints. """

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, output, target):
        abs_diff = torch.abs(output - target)
        quadratic = torch.min(abs_diff, torch.tensor(self.delta))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear
        return loss.mean()


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        loss = torch.mean(torch.clamp(self.margin - target * (output1 - output2), min=0))
        return loss


"""Focal loss is designed to address class imbalance by focusing more on hard-to-classify examples.
This can be beneficial if your ordinal labels are imbalanced or if you want the model to focus on making precise distinctions."""


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        mse_loss = F.mse_loss(outputs, targets, reduction='none')
        pt = torch.exp(-mse_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * mse_loss
        return F_loss.mean()


"""Smooth L1 Loss, also known as Huber Loss, combines the benefits of both L1 and L2 losses.
It is less sensitive to outliers than MSE and can handle minor discrepancies better."""


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        return self.smooth_l1_loss(outputs, targets)


"""Using MSE with label smoothing can help the model generalize better by softening the target labels,
which can make the model less confident and better at distinguishing between close labels."""


class MSEWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(MSEWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        targets = targets.float() * (1 - self.smoothing) + self.smoothing / 2
        return self.mse_loss(outputs, targets)


"""This custom loss function can be implemented to better capture the ordinal nature
of the labels by penalizing misclassifications based on their distance from the true label."""

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        targets = targets.long()
        log_probs = F.log_softmax(outputs, dim=1)
        ordinal_targets = torch.zeros_like(log_probs).scatter_(1, targets.view(-1, 1), 1)
        ordinal_targets[:, 1:] = ordinal_targets[:, 1:].cumsum(dim=1)[:, :-1]
        loss = -(ordinal_targets * log_probs).sum(dim=1).mean()
        return loss
