<<<<<<< HEAD
from datasets.helmms_dataset import HELMMSPalmVeinDataset

ROOT = r"data/raw/HELM-MS"


ds = HELMMSPalmVeinDataset(
    root_dir=ROOT,
    dataset_name="CASIA-Pure",
    split="train",
    band="850",
    img_size=224
)

print("Train samples:", len(ds))
print("Classes:", ds.num_classes)

raw, clahe, label = ds[0]
print("Raw shape:", raw.shape)
print("CLAHE shape:", clahe.shape)
print("Label:", label)
=======
from datasets.helmms_dataset import HELMMSPalmVeinDataset

ROOT = r"data/raw/HELM-MS"


ds = HELMMSPalmVeinDataset(
    root_dir=ROOT,
    dataset_name="CASIA-Pure",
    split="train",
    band="850",
    img_size=224
)

print("Train samples:", len(ds))
print("Classes:", ds.num_classes)

raw, clahe, label = ds[0]
print("Raw shape:", raw.shape)
print("CLAHE shape:", clahe.shape)
print("Label:", label)
>>>>>>> 793e70482332d6cabced4d75d1a53ed9a9c1a2f6
