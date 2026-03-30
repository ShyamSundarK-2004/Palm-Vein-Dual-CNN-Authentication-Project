import torch
import time
import numpy as np
import random
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_model(model, loader, device, model_type, max_pairs=20000):

    model.eval()

    features = []
    labels = []

    total_time = 0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for raw, clahe, y in loader:

            raw = raw.to(device)
            clahe = clahe.to(device)
            y = y.to(device)

            start = time.time()

            # 🔥 FEATURE EXTRACTION
            if model_type == "raw":
                feat = model.extract_features(raw)
                outputs = model(raw)
            else:
                feat = model.extract_features(raw, clahe)
                outputs = model(raw, clahe)

            # 🔥 Handle ECG (tuple output)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            end = time.time()

            total_time += (end - start)
            total_samples += raw.size(0)

            # 🔥 Accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()

            features.append(feat.cpu().numpy())
            labels.append(y.cpu().numpy())

    # ===============================
    # 🔥 STACK FEATURES
    # ===============================
    features = np.vstack(features)
    labels = np.hstack(labels)

    # 🔥 IMPORTANT: Normalize features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # ===============================
    # 🔥 BALANCED PAIR GENERATION
    # ===============================
    genuine_scores = []
    impostor_scores = []

    n = len(features)
    indices = list(range(n))

    # 🔥 Genuine pairs
    for _ in range(max_pairs // 2):
        while True:
            i = random.choice(indices)
            j = random.choice(indices)
            if i != j and labels[i] == labels[j]:
                break

        sim = cosine_similarity(
            features[i].reshape(1, -1),
            features[j].reshape(1, -1)
        )[0][0]

        genuine_scores.append(sim)

    # 🔥 Impostor pairs
    for _ in range(max_pairs // 2):
        while True:
            i = random.choice(indices)
            j = random.choice(indices)
            if labels[i] != labels[j]:
                break

        sim = cosine_similarity(
            features[i].reshape(1, -1),
            features[j].reshape(1, -1)
        )[0][0]

        impostor_scores.append(sim)

    scores = np.array(genuine_scores + impostor_scores)
    pair_labels = np.array([1]*len(genuine_scores) + [0]*len(impostor_scores))

    # ===============================
    # 🔥 ROC + EER
    # ===============================
    fpr, tpr, thresholds = roc_curve(pair_labels, scores)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]

    eer_threshold = thresholds[eer_idx]

    FAR = np.mean((scores >= eer_threshold) & (pair_labels == 0))
    FRR = np.mean((scores < eer_threshold) & (pair_labels == 1))

    accuracy = correct / total_samples
    inference_time = (total_time / total_samples) * 1000  # ms

    return {
        "Accuracy": accuracy,
        "FAR": FAR,
        "FRR": FRR,
        "EER": eer,
        "Inference Time (ms)": inference_time
    }