import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# Hybrid Loss combining Focal Loss and Dice Loss
class HybridLoss(nn.Module):
    def __init__(self, num_classes, lambda1=0.5, lambda2=0.5):
        super(HybridLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets, alpha, gamma):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (alpha[targets] * (1 - p_t) ** gamma[targets] * ce_loss).mean()
        dice_loss = self.dice_loss(inputs, targets)
        return self.lambda1 * focal_loss + self.lambda2 * dice_loss
