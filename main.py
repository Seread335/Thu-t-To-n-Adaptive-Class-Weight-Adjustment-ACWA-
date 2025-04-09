import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
from acwa_trainer import create_imbalanced_cifar10
import numpy as np

# Focal Loss with Label Smoothing
class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, outputs, labels):
        logpt = nn.functional.cross_entropy(outputs, labels, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        return loss.mean()

# Warm-up Scheduler
class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            cosine_epoch = self.last_epoch - self.warmup_epochs
            cosine_max_epochs = self.max_epochs - self.warmup_epochs
            return [self.eta_min + (base_lr - self.eta_min) * (1 + torch.cos(torch.tensor(cosine_epoch * torch.pi / cosine_max_epochs))) / 2 for base_lr in self.base_lrs]

# Enhanced SimpleCNN
class EnhancedSimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Hybrid Loss
class HybridLoss(nn.Module):
    def __init__(self, num_classes, lambda1=0.7, lambda2=0.3):
        super(HybridLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets, alpha, gamma):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        focal_loss = (alpha[targets] * (1 - p_t) ** gamma[targets] * ce_loss).mean()
        dice_loss = self.dice_loss(inputs, targets)
        return self.lambda1 * focal_loss + self.lambda2 * dice_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_one_hot = nn.functional.one_hot(targets, num_classes=inputs.shape[1]).float()
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# EfficientNet Model
class ImprovedModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedModel, self).__init__()
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        self.base_model.classifier[1] = nn.Linear(1536, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Updated ACWATrainer
class ACWATrainer:
    def __init__(self, num_classes, alpha=0.01, beta=0.95, f1_target_start=0.6, update_freq=100):
        self.weights = torch.ones(num_classes).to(device)
        self.alpha_base = alpha
        self.beta = beta
        self.f1_target_start = f1_target_start
        self.f1_target = f1_target_start
        self.update_freq = update_freq
        self.batch_count = 0
        self.start_reweight_epoch = 20

    def get_weighted_loss(self, outputs, labels, alpha, gamma):
        criterion = HybridLoss(num_classes=10)
        return criterion(outputs, labels, alpha, gamma)

    def update_weights(self, outputs, labels, epoch):
        if epoch < self.start_reweight_epoch:
            return
        self.batch_count += 1
        if self.batch_count % self.update_freq == 0:
            _, preds = torch.max(outputs, 1)
            f1_per_class = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average=None, zero_division=0)
            f1 = f1_per_class.mean()
            self.f1_target = min(0.9, self.f1_target_start + 0.002 * (epoch - self.start_reweight_epoch))
            alpha = self.alpha_base * (1 - f1 / self.f1_target)
            error = self.f1_target - f1
            self.weights = self.beta * self.weights + (1 - self.beta) * (self.weights + alpha * error)
            self.weights = torch.clamp(self.weights, 0.5, 2.0)

if __name__ == '__main__':
    torch.manual_seed(42)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset, testset, sampler = create_imbalanced_cifar10(imbalance_ratio=0.1)  # Updated to unpack three values
    trainloader = DataLoader(trainset, batch_size=64, sampler=sampler, num_workers=0)  # Use sampler
    valloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)  # Use testset as validation set
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedModel(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    acwa_trainer = ACWATrainer(num_classes=10)

    loss_history = []
    f1_history = []
    num_epochs = 150
    patience = 40
    best_f1 = 0.0
    patience_counter = 0
    smoothed_f1 = 0.0
    beta_ema = 0.9

    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            alpha = torch.ones(10, device=device)
            gamma = torch.ones(10, device=device) * 2
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = acwa_trainer.get_weighted_loss(outputs, labels, alpha, gamma)
                loss.backward()
                optimizer.step()
                acwa_trainer.update_weights(outputs, labels, epoch)
                running_loss += loss.item()
            epoch_loss = running_loss / len(trainloader)
            loss_history.append(epoch_loss)

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in valloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
            val_f1 = np.mean(f1_per_class)
            smoothed_f1 = beta_ema * smoothed_f1 + (1 - beta_ema) * val_f1
            for c in range(10):
                alpha[c] = 1 / (1 + np.exp(f1_per_class[c] - 0.5))
                gamma[c] = 2 + 4 * (1 - f1_per_class[c])
            f1_history.append(smoothed_f1)
            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Smoothed Val F1 = {smoothed_f1:.4f}")

            scheduler.step()
            if smoothed_f1 > best_f1:
                best_f1 = smoothed_f1
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best model saved at epoch {epoch+1} with Smoothed F1 = {smoothed_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at Epoch {epoch+1}")
                    break
    except KeyboardInterrupt:
        print("Training interrupted by user")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(f1_history, label='Validation Macro F1 (Smoothed)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.title('Enhanced ACWA on CIFAR-10 Imbalanced')
    plt.show()
