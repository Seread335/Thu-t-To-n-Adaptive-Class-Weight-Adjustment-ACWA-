import torch
import torch.optim as optim
from acwa_trainer import SimpleCNN, ACWATrainer, create_imbalanced_cifar10

def main():
    # Chuẩn bị dữ liệu
    trainset, testset = create_imbalanced_cifar10(imbalance_ratio=0.1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Split validation set from trainset
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)

    # Khởi tạo mô hình và thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Calculate class frequencies
    class_counts = torch.bincount(torch.tensor(trainset.dataset.targets))
    class_frequencies = class_counts.float() / class_counts.sum()

    # Initialize ACWA Trainer with class frequencies
    acwa_trainer = ACWATrainer(
        model=model,
        num_classes=10,
        alpha=0.02,
        beta=0.9,
        target_f1=0.8,
        update_freq=50,
        class_frequencies=class_frequencies
    )

    # Huấn luyện
    num_epochs = 10
    best_f1 = 0
    early_stop_counter = 0
    patience = 5  # Stop training if no improvement for 5 epochs

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

            # Cập nhật metrics
            acwa_trainer.update_metrics(outputs, labels)

            # Định kỳ cập nhật weights
            if i % acwa_trainer.update_freq == acwa_trainer.update_freq - 1:
                acwa_trainer.update_weights()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}")

        # Evaluate on validation set for early stopping
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}")

        # Early stopping logic
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()