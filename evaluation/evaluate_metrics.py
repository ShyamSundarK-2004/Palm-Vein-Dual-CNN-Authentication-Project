import torch
import time
import numpy as np
from sklearn.metrics import roc_curve


def evaluate_model(model, loader, device, model_type):

    model.eval()
    all_scores = []
    all_labels = []

    total_time = 0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for raw, clahe, labels in loader:

            raw, clahe, labels = raw.to(device), clahe.to(device), labels.to(device)

            start = time.time()

            if model_type == "raw":
                outputs = model(raw)
            elif model_type == "clahe":
                outputs = model(clahe)
            else:
                outputs = model(raw, clahe)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            end = time.time()

            total_time += (end - start)
            total_samples += raw.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            correct += (preds == labels).sum().item()

            # 🔥 confidence of predicted class
            confidence = probs.max(dim=1)[0]

            # 🔥 correct = genuine (1), wrong = impostor (0)
            match = (preds == labels).int()

            all_scores.extend(confidence.cpu().numpy())
            all_labels.extend(match.cpu().numpy())

    accuracy = correct / total_samples
    inference_time = (total_time / total_samples) * 1000  # ms

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # ROC
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fnr = 1 - tpr

    # EER
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]

    # FAR & FRR at EER threshold
    threshold = thresholds[idx]

    FAR = np.mean((all_scores >= threshold) & (all_labels == 0))
    FRR = np.mean((all_scores < threshold) & (all_labels == 1))

    return {
        "Accuracy": accuracy,
        "FAR": FAR,
        "FRR": FRR,
        "EER": eer,
        "Inference Time (ms)": inference_time
    }