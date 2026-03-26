import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.ecg_model import ECGFusionModel   # 🔥 NEW MODEL


# 🔥 Transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(128, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss, correct, total = 0, 0, 0

    for raw, clahe, labels in tqdm(loader, desc="Train"):

        raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

        optimizer.zero_grad()

        result = model(raw, clahe)

        # 🔥 handle (output, gate)
        if isinstance(result, tuple):
            outputs = result[0]
        else:
            outputs = result

        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item() * raw.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += raw.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()
    correct, total = 0, 0

    for raw, clahe, labels in tqdm(loader, desc="Test"):

        raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

        result = model(raw, clahe)

        if isinstance(result, tuple):
            outputs = result[0]
        else:
            outputs = result

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += raw.size(0)

    return correct / total


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🔥 Device:", device)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    train_ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "train", BAND, 128, transform=train_transform)
    test_ds  = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "test", BAND, 128, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = ECGFusionModel(num_classes=train_ds.num_classes).to(device)

    # 🔥 Improved Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 🔥 Better optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # 🔥 Best scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    EPOCHS = 100
    best_acc = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)

        scheduler.step()

        print(f"[ECG FUSION] Epoch {epoch}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc*100:.2f}% | "
              f"Test: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "outputs/ecg_fusion_model.pth")

    print("🔥 Best ECG Accuracy:", best_acc)


if __name__ == "__main__":
    main()