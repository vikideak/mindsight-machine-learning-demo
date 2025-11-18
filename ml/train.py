import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets, transforms, models
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json

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
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return accuracy, precision, recall, f1


def main():
    t0 = time.time()
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    full_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_data = filter_images(full_train)
    test_data = filter_images(full_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_CLASSES))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
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

        print(f"Epoch {epoch+1}/3 - Loss: {total_loss:.4f}")

    # Evaluate with extended metrics
    accuracy, precision, recall, f1 = compute_metrics(model, test_loader, device)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    torch.save(model.state_dict(), "ml/model.pth")

    with open("ml/labels.json", "w") as f:
        json.dump({i: c for i, c in enumerate(TARGET_CLASSES)}, f)

    print("Training complete. Model saved to ml/model.pth")
    print("Time:", time.time()-t0)

if __name__ == "__main__":
    main()
