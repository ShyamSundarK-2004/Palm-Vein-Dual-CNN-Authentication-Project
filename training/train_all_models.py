import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms

from torch.cuda.amp import autocast, GradScaler

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.raw_cnn import RawResNet18
from models.dual_fusion import ConcatFusionModel
from models.dual_sum import SumFusionModel
from models.ecg_model import ECGFusionModel

# 🔥 SEED
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# 🔥 TRANSFORMS (STRONG FOR GENERALIZATION)
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 🔥 FORWARD
def forward_pass(model, raw, clahe, model_type):
    if model_type == "raw":
        return model(raw)
    elif model_type == "clahe":
        return model(clahe)
    else:
        return model(raw, clahe)

def extract_output(result):
    return result[0] if isinstance(result, tuple) else result

# 🔥 TRAIN ONE EPOCH
def train_one_epoch(model, loader, optimizer, criterion, device, model_type, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for raw, clahe, labels in tqdm(loader, desc="Train"):
        raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = extract_output(forward_pass(model, raw, clahe, model_type))
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * raw.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += raw.size(0)

    return total_loss / total, correct / total

# 🔥 EVALUATE
@torch.no_grad()
def evaluate(model, loader, device, model_type):
    model.eval()
    correct, total = 0, 0

    for raw, clahe, labels in tqdm(loader, desc="Test"):
        raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

        outputs = extract_output(forward_pass(model, raw, clahe, model_type))
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += raw.size(0)

    return correct / total

# 🔥 TRAIN MODEL (FINAL)
def train_model(model, name, model_type, train_loader, test_loader, device):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)

    scaler = GradScaler()

    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(1, 21):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, model_type, scaler
        )

        test_acc = evaluate(model, test_loader, device, model_type)

        print(f"[{name}] Epoch {epoch}/20 | "
              f"Loss: {train_loss:.4f} | "
              f"Train: {train_acc*100:.2f}% | "
              f"Test: {test_acc*100:.2f}%")

        # 🔥 EARLY STOPPING + BEST SAVE
        if test_acc > best_acc:
            best_acc = test_acc
            counter = 0
            torch.save(model.state_dict(), f"outputs/{name}.pth")
        else:
            counter += 1

        if counter >= patience:
            print("🛑 Early stopping triggered")
            break

    print(f"🔥 Best {name} Accuracy: {best_acc*100:.2f}%")

# 🔥 MAIN
def run():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Using device:", device)

    ROOT = r"data/raw/HELM-MS"

    # ✅ TRAIN → PolyU
    train_ds = HELMMSPalmVeinDataset(
        ROOT, "PolyU-Pure", "850", 224, transform=train_transform
    )

    # ✅ TEST → CASIA
    test_ds = HELMMSPalmVeinDataset(
        ROOT, "CASIA-Pure", "850", 224, transform=test_transform
    )

    print("Train size:", len(train_ds))
    print("Test size:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    num_classes = train_ds.num_classes

    # 🔥 TRAIN ALL MODELS
    train_model(RawResNet18(num_classes), "raw", "raw", train_loader, test_loader, device)
    train_model(RawResNet18(num_classes), "clahe", "clahe", train_loader, test_loader, device)
    train_model(ConcatFusionModel(num_classes), "concat", "dual", train_loader, test_loader, device)
    train_model(SumFusionModel(num_classes), "sum", "dual", train_loader, test_loader, device)
    train_model(ECGFusionModel(num_classes), "ecg", "dual", train_loader, test_loader, device)

if __name__ == "__main__":
    run()