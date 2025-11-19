import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json
from ml.transforms import get_train_transform, get_transform

TARGET_CLASSES = ["airplane", "automobile", "truck"]

def filter_images(dataset): 
    keep = [dataset.class_to_idx[c] for c in TARGET_CLASSES]

    indices = [i for i, t in enumerate(dataset.targets) if t in keep]

    dataset.data = dataset.data[indices]
    dataset.targets = [dataset.targets[i] for i in indices]

    mapping = {old: i for i, old in enumerate(keep)}
    dataset.targets = [mapping[t] for t in dataset.targets]

    print(f"Filtered dataset to {len(dataset)} images.")
    return dataset


def compute_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, precision, recall, f1


def main():

    train_transform = get_train_transform()
    test_transform = get_transform()

    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    full_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    full_train = filter_images(full_train)
    full_test = filter_images(full_test)

    val_ratio = 0.1
    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train_data, val_data = torch.utils.data.random_split(full_train, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
    test_loader = torch.utils.data.DataLoader(full_test, batch_size=128, shuffle=False)


    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_CLASSES))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    for epoch in range(4):
        model.train()
        total_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc, val_prec, val_rec, val_f1 = compute_metrics(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/4")
        print(f"Train Loss: {total_loss:.4f}")
        print(f"Val Acc:    {val_acc:.4f}")
        print(f"Val Prec:   {val_prec:.4f}")
        print(f"Val Rec:    {val_rec:.4f}")
        print(f"Val F1:     {val_f1:.4f}")


    test_acc, test_prec, test_rec, test_f1 = compute_metrics(model, test_loader, device)

    print("\n=== Final Test Metrics ===")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")


    torch.save(model.state_dict(), "ml/model.pth")

    with open("ml/labels.json", "w") as f:
        json.dump({i: c for i, c in enumerate(TARGET_CLASSES)}, f)

    print("Training complete. Model saved to ml/model.pth")


if __name__ == "__main__":
    main()
