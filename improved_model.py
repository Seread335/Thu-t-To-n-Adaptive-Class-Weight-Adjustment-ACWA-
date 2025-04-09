import torch
import torch.nn as nn
from torchvision import models

# Improved Model with EfficientNet-B3
class ImprovedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedModel, self).__init__()
        self.base_model = models.efficientnet_b3(pretrained=True)
        self.fc = nn.Linear(1536, num_classes)  # EfficientNet-B3 output

    def forward(self, x):
        x = self.base_model.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.fc(x)
        return x
