import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from evaluation.evaluate_metrics import evaluate_model

from models.raw_cnn import RawResNet18
from models.dual_fusion import ConcatFusionModel
from models.ecg_model import ECGFusionModel


device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 TRANSFORM
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

ROOT = r"data/raw/HELM-MS"

# 🔥 DATASETS
casia_test = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "850", 128, transform=transform)
polyu_test = HELMMSPalmVeinDataset(ROOT, "PolyU-Pure", "850", 128, transform=transform)

casia_loader = DataLoader(casia_test, batch_size=32, shuffle=False)
polyu_loader = DataLoader(polyu_test, batch_size=32, shuffle=False)

# 🔥 MODELS
models = {
    "CASIA_RAW": (RawResNet18(casia_test.num_classes), "outputs/CASIA-Pure_raw.pth", "raw"),
    "CASIA_CONCAT": (ConcatFusionModel(casia_test.num_classes), "outputs/CASIA-Pure_concat.pth", "dual"),
    "CASIA_ECG": (ECGFusionModel(casia_test.num_classes), "outputs/CASIA-Pure_ecg.pth", "dual"),

    "POLYU_ECG": (ECGFusionModel(polyu_test.num_classes), "outputs/PolyU-Pure_ecg.pth", "dual"),
}


# 🔥 RUN FUNCTION
def run_eval(name, model, path, model_type, loader):
    print(f"\n🚀 Evaluating {name}...")

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

    metrics = evaluate_model(model, loader, device, model_type, max_pairs=15000)

    print(f"\n📊 {name} RESULTS:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


print("\n🔥 ===== FINAL RESULTS ===== 🔥\n")

# =============================
# ✅ SAME DATASET
# =============================
run_eval("CASIA RAW → CASIA", *models["CASIA_RAW"], casia_loader)
run_eval("CASIA CONCAT → CASIA", *models["CASIA_CONCAT"], casia_loader)
run_eval("CASIA ECG → CASIA", *models["CASIA_ECG"], casia_loader)

run_eval("POLYU ECG → POLYU", *models["POLYU_ECG"], polyu_loader)

# =============================
# 🔥 CROSS DATASET
# =============================
run_eval("CASIA ECG → POLYU", *models["CASIA_ECG"], polyu_loader)
run_eval("POLYU ECG → CASIA", *models["POLYU_ECG"], casia_loader)