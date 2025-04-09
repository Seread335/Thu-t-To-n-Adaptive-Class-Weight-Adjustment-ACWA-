import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict

try:
    from torchmetrics import F1Score
except ImportError:
    raise ImportError("The 'torchmetrics' library is required. Install it using 'pip install torchmetrics'.")

# Updated Focal Loss with Class-wise Gamma and Dynamic Alpha
class FocalLoss(nn.Module):
    def __init__(self, gamma_dict, alpha_dict=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma_dict = gamma_dict
        self.alpha_dict = alpha_dict
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-ce_loss)
        gamma = torch.tensor([self.gamma_dict[t.item()] for t in targets], device=inputs.device)
        loss = (1 - p_t) ** gamma * ce_loss
        if self.alpha_dict is not None:
            alpha_t = torch.tensor([self.alpha_dict[t.item()] for t in targets], device=inputs.device)
            loss = alpha_t * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# Updated create_imbalanced_cifar10 with SMOTE and WeightedRandomSampler
def create_imbalanced_cifar10(imbalance_ratio=0.1):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Tải tập train và test
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Tạo imbalance bằng cách giảm số lượng mẫu của một số lớp
    targets = np.array(full_trainset.targets)
    class_counts = defaultdict(int)
    
    # Chọn 3 lớp để làm minority (0, 1, 2)
    minority_classes = [0, 1, 2]
    
    # Tạo mask để lọc dữ liệu
    mask = np.ones(len(targets), dtype=bool)
    for class_idx in range(10):
        class_mask = (targets == class_idx)
        if class_idx in minority_classes:
            # Giữ lại chỉ một phần nhỏ samples cho minority classes
            keep_prob = imbalance_ratio
            keep_indices = np.where(class_mask)[0]
            np.random.shuffle(keep_indices)
            keep_count = int(len(keep_indices) * keep_prob)
            mask[keep_indices[keep_count:]] = False
    
    # Áp dụng mask
    imbalanced_trainset = torch.utils.data.Subset(full_trainset, np.where(mask)[0])

    # WeightedRandomSampler
    class_counts = np.bincount(targets[mask])
    class_weights = 1.0 / (class_counts + 1e-7)
    sample_weights = class_weights[targets[mask]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    return imbalanced_trainset, testset, sampler

# Updated SimpleCNN with Residual Connections and Multi-Head Attention
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
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
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

# 3. Triển khai ACWA
class ACWATrainer:
    def __init__(self, model, num_classes, alpha=0.01, beta=0.9, target_f1=0.9, update_freq=100, class_frequencies=None):
        self.model = model
        self.num_classes = num_classes
        self.alpha = alpha  # learning rate cho weight adjustment
        self.beta = beta    # smoothing factor
        self.target_f1 = target_f1
        self.update_freq = update_freq  # cập nhật sau mỗi bao nhiêu batch
        
        # Khởi tạo weights
        device = next(model.parameters()).device  # Automatically detect device
        self.f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="none").to(device)
        
        # Initialize weights with epsilon for numerical stability
        if class_frequencies:
            self.weights = torch.tensor([1.0 / (freq + 1e-7) for freq in class_frequencies], device=device)
        else:
            self.weights = torch.ones(num_classes, device=device)
        self.class_f1 = torch.zeros(num_classes, device=device)
        self.class_counts = torch.zeros(num_classes, device=device)
        
    def reset_metrics(self):
        device = self.weights.device  # Use the same device as weights
        self.class_f1 = torch.zeros(self.num_classes, device=device)
        self.class_counts = torch.zeros(self.num_classes, device=device)
        
    def update_weights(self):
        # Compute F1 scores from accumulated metrics
        f1_scores = self.f1_metric.compute()
        self.f1_metric.reset()  # Reset after computing F1 scores
        
        # Cập nhật weights
        for c in range(self.num_classes):
            if self.class_counts[c] > 0:  # Only update if the class appears in the batch
                error_c = self.target_f1 - f1_scores[c]
                delta = self.alpha * error_c
                
                # Áp dụng smoothing
                new_weight = self.weights[c] + delta
                smoothed_weight = self.beta * self.weights[c] + (1 - self.beta) * new_weight
                
                # Giới hạn weight trong khoảng [0.5, 2.0]
                self.weights[c] = torch.clamp(smoothed_weight, 0.5, 2.0)
        
        # Reset metrics sau mỗi lần cập nhật
        self.reset_metrics()
        
    def get_weighted_loss(self, outputs, labels):
        # Tính loss với trọng số hiện tại
        loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
        
        # Áp dụng weights
        weighted_loss = torch.zeros_like(loss)
        for c in range(self.num_classes):
            class_mask = (labels == c)
            weighted_loss[class_mask] = loss[class_mask] * self.weights[c]
            
        return weighted_loss.mean()
    
    def update_metrics(self, outputs, labels):
        # Only update metrics, defer computation to update_weights
        self.f1_metric.update(outputs, labels)

# 4. Hàm huấn luyện
def train_with_acwa():
    # Chuẩn bị dữ liệu
    imbalanced_trainset, testset, sampler = create_imbalanced_cifar10(imbalance_ratio=0.1)
    
    # Chia tập validation (20% của tập train)
    train_size = int(0.8 * len(imbalanced_trainset))
    val_size = len(imbalanced_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(imbalanced_trainset, [train_size, val_size])
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, sampler=sampler, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedSimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Khởi tạo ACWA trainer
    acwa_trainer = ACWATrainer(model, num_classes=10, alpha=0.02, beta=0.9, target_f1=0.8, update_freq=50)
    
    # Theo dõi weights qua các epoch
    weight_history = []
    
    # Huấn luyện
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = acwa_trainer.get_weighted_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Cập nhật metrics cho ACWA
            acwa_trainer.update_metrics(outputs, labels)
            
            # Định kỳ cập nhật weights
            if i % acwa_trainer.update_freq == acwa_trainer.update_freq - 1:
                acwa_trainer.update_weights()
                weight_history.append(acwa_trainer.weights.detach().cpu().numpy().copy())
            
            running_loss += loss.item()
        
        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = acwa_trainer.get_weighted_loss(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Tính các chỉ số
        val_loss /= len(valloader)
        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {running_loss/len(trainloader):.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, '
              f'Val F1: {val_f1:.4f}')
    
    # Vẽ biểu đồ weight history
    weight_history = np.array(weight_history)
    plt.figure(figsize=(12, 6))
    for c in range(10):
        plt.plot(weight_history[:, c], label=f'Class {c}')
    plt.title('Class Weight Adjustment Over Training')
    plt.xlabel('Update Steps')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Đánh giá trên tập test
    model.eval()
    test_acc = 0.0
    test_f1 = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Final Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}')

if __name__ == '__main__':
    train_with_acwa()
