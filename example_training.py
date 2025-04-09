import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
from improved_model import ImprovedModel
from hybrid_loss import HybridLoss
from acwa_trainer import create_imbalanced_cifar10

def main():
    # Prepare data
    trainset, testset, sampler = create_imbalanced_cifar10(imbalance_ratio=0.1)
    trainloader = DataLoader(trainset, batch_size=64, sampler=sampler, num_workers=2)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedModel(num_classes=10).to(device)
    criterion = HybridLoss(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

    # Training loop
    num_epochs = 100
    alpha = torch.ones(10, device=device)
    gamma = torch.ones(10, device=device) * 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, alpha, gamma)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation and Dynamic Adjustment
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        for c in range(10):
            alpha[c] = 1 / (1 + np.exp(f1_per_class[c] - 0.5))
            gamma[c] = 2 + 4 * (1 - f1_per_class[c])

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}, F1 Macro: {np.mean(f1_per_class):.4f}")
        scheduler.step()

if __name__ == "__main__":
    main()
