import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# Assuming these are your local imports
from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.raw_cnn import RawResNet18  # Your original single-input model
from models.dual_fusion import ConcatFusionModel 
from models.ecg_model import ECGFusionModel  # Your new models

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# --- TRANSFORMS ---
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- DATASET WRAPPER (Handles Dual Input for Fusion) ---
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # HELMMS returns (raw, clahe, label)
        raw, clahe, label = self.subset[idx] 
        if self.transform:
            raw = self.transform(raw)
            clahe = self.transform(clahe)
        return raw, clahe, label

# --- ENGINE ---
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, model_type):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for raw, enh, labels in tqdm(loader, desc=f"Training {model_type}", leave=False):
        raw, enh, labels = raw.to(device), enh.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            # Logic to handle different input requirements
            if model_type == "RAW":
                outputs = model(raw)
            else:
                outputs = model(raw, enh)
                
            # Unpack ECGFusion output if necessary
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * raw.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += raw.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device, model_type):
    model.eval()
    correct, total = 0, 0
    for raw, enh, labels in loader:
        raw, enh, labels = raw.to(device), enh.to(device), labels.to(device)
        
        if model_type == "RAW":
            outputs = model(raw)
        else:
            outputs = model(raw, enh)
            
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += raw.size(0)
    return correct / total if total > 0 else 0

# --- MAIN ---
def run(model_type="RAW"): # Choices: "RAW", "ECG", "CONCAT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*30)
    print(f"STARTING TRAINING: {model_type}")
    print(f"="*30)

    ROOT = "data/raw/HELM-MS"
    full_ds = HELMMSPalmVeinDataset(ROOT, "PolyU-Pure", "850", 224, transform=None)
    
    train_indices, val_indices, test_indices = random_split(
        full_ds, [int(0.8*len(full_ds)), int(0.1*len(full_ds)), len(full_ds)-int(0.8*len(full_ds))-int(0.1*len(full_ds))],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(TransformedSubset(train_indices, train_transform), batch_size=16, shuffle=True)
    val_loader   = DataLoader(TransformedSubset(val_indices, val_transform), batch_size=16, shuffle=False)
    test_loader  = DataLoader(TransformedSubset(test_indices, val_transform), batch_size=16, shuffle=False)

    # Model Selection
    num_classes = full_ds.num_classes
    if model_type == "RAW":
        model = RawResNet18(num_classes).to(device)
    elif model_type == "ECG":
        model = ECGFusionModel(num_classes).to(device)
    elif model_type == "CONCAT":
        model = ConcatFusionModel(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(1, 11):
        loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, model_type)
        val_acc = evaluate(model, val_loader, device, model_type)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Train: {train_acc*100:.1f}% | Val: {val_acc*100:.1f}%")

    test_acc = evaluate(model, test_loader, device, model_type)
    print(f"Final {model_type} Test Accuracy: {test_acc*100:.2f}%")
    
    # Save the model
    save_path = f"{model_type}_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # This will train all three models sequentially
    models_to_train = ["ECG", "CONCAT"]
    
    for m in models_to_train:
        run(model_type=m)