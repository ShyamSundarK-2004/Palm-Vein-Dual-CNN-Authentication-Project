import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.ecg_fusion import ECGFusionVector


def train_one_epoch(model, loader, optimizer, criterion, device):

    model.train()
    total_loss, correct, total = 0, 0, 0

    for raw, clahe, labels in tqdm(loader, desc="Train", leave=False):

        raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, gate_vals = model(raw, clahe)

        loss = criterion(outputs, labels)

        loss.backward()

        # gradient clipping
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

    for raw, clahe, labels in tqdm(loader, desc="Test", leave=False):

        raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

        outputs, gate_vals = model(raw, clahe)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += raw.size(0)

    return correct / total


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    train_ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "train", BAND, 128)
    test_ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "test", BAND, 128)

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )

    model = ECGFusionVector(num_classes=train_ds.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    EPOCHS = 100
    best_acc = 0

    for epoch in range(1, EPOCHS + 1):

        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_acc = evaluate(model, test_loader, device)

        scheduler.step()

        t1 = time.time()

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | "
              f"Test Acc: {test_acc*100:.2f}% | "
              f"Time: {t1-t0:.1f}s")

        # save best model
        if test_acc > best_acc:

            best_acc = test_acc

            torch.save(
                model.state_dict(),
                "outputs/best_ecg_vector_casia.pth"
            )

    print("Best Accuracy:", best_acc)


if __name__ == "__main__":
    main()