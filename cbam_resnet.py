import torch
import torch.nn as nn
from torchvision import models

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = torch.sigmoid(self.spatial_attention(torch.cat([ca.mean(dim=1, keepdim=True), ca.max(dim=1, keepdim=True)[0]], dim=1))) * ca
        return sa

# CBAM-ResNet Model
class CBAMResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CBAMResNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        self.cbam = CBAM(2048)  # CBAM module
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.cbam(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.fc(x)
        return x
