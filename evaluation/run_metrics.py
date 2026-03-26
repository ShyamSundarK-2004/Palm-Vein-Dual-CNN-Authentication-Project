import torch
from torch.utils.data import DataLoader

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from evaluation.evaluate_metrics import evaluate_model

from models.raw_cnn import RawResNet18
from models.dual_fusion import ConcatFusionModel
from models.dual_sum import SumFusionModel
from models.ecg_model import ECGFusionModel

import torchvision.transforms as transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Transform
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ROOT = r"data/raw/HELM-MS"

# 🔥 Dataset
test_ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "test", "850", 128, transform=test_transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)


# 🔥 MODEL CONFIG
models = [
    ("RAW", RawResNet18(test_ds.num_classes), "outputs/CASIA-Pure_raw.pth", "raw"),
    ("CLAHE", RawResNet18(test_ds.num_classes), "outputs/CASIA-Pure_clahe.pth", "clahe"),
    ("CONCAT", ConcatFusionModel(test_ds.num_classes), "outputs/CASIA-Pure_concat.pth", "dual"),
    ("SUM", SumFusionModel(test_ds.num_classes), "outputs/CASIA-Pure_sum.pth", "dual"),
    ("ECG", ECGFusionModel(test_ds.num_classes), "outputs/CASIA-Pure_ecg.pth", "dual"),
]


print("\n🔥 ===== CASIA RESULTS (ALL MODELS) ===== 🔥\n")

for name, model, path, model_type in models:

    print(f"\n🚀 Evaluating {name}...")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    metrics = evaluate_model(model, test_loader, device, model_type)

    print(f"\n📊 {name} RESULTS:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")