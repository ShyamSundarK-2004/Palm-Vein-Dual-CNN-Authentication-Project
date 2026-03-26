import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from sklearn.metrics import roc_curve
import torchvision.transforms as transforms

# Import your project modules
from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.ecg_model import ECGFusionModel

# 1. SETTINGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "data/raw/HELM-MS"
TRAINED_MODEL_PATH = "outputs/PolyU-Pure_ecg.pth" # Your PolyU weights
TEST_DATASET_NAME = "CASIA-Pure"                 # The new dataset

# 2. TRANSFORMS (Must match what you used in training)
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 3. METRIC FUNCTION
def calculate_metrics(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    total_samples, correct = 0, 0

    print(f"--- Starting Evaluation on {TEST_DATASET_NAME} ---")
    with torch.no_grad():
        for raw, clahe, labels in loader:
            raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

            # Forward pass
            outputs = model(raw, clahe)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            # Identification Accuracy
            correct += (preds == labels).sum().item()
            total_samples += raw.size(0)

            # For Biometric EER (Genuine vs Imposter scores)
            batch_probs = probs.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            for i in range(len(batch_labels)):
                # Score for the correct identity (Genuine)
                all_scores.append(batch_probs[i, batch_labels[i]])
                all_labels.append(1)
                
                # Max score for any wrong identity (Imposter)
                wrong_scores = np.delete(batch_probs[i], batch_labels[i])
                all_scores.append(np.max(wrong_scores))
                all_labels.append(0)

    # Math logic
    accuracy = correct / total_samples
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # ROC / EER
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    
    eer = (fpr[idx] + fnr[idx]) / 2
    far = fpr[idx]
    frr = fnr[idx]

    return accuracy, far, frr, eer

# 4. EXECUTION
def main():
    # Load CASIA to get its properties, but we use PolyU's class count for the model
    # Note: If CASIA has more classes than PolyU, this might throw an error.
    # Usually, we use num_classes from the TRAINED model.
    
    # Assuming PolyU had 100 classes during training:
    NUM_CLASSES_TRAINED = 500 
    
    model = ECGFusionModel(NUM_CLASSES_TRAINED).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
    print(f"✅ Weights Loaded: {TRAINED_MODEL_PATH}")

    # Load CASIA Data
    test_ds = HELMMSPalmVeinDataset(ROOT, TEST_DATASET_NAME, "850", 128, transform=test_transform)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    # Run Eval
    acc, far, frr, eer = calculate_metrics(model, loader, device)

    print("\n" + "="*30)
    print(f"CROSS-DATASET RESULTS")
    print(f"Train: PolyU | Test: {TEST_DATASET_NAME}")
    print("="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"EER:      {eer:.4f}")
    print(f"FAR:      {far:.4f}")
    print(f"FRR:      {frr:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()