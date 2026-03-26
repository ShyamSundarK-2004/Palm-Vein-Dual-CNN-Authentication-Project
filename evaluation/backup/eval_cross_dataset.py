import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("/content/drive/MyDrive/Palm_Vein_Auth_Project/palm-vein-dual-cnn-auth")
from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.ecg_embedder import ECGScalarEmbedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_far_frr_eer(genuine_scores, impostor_scores, steps=2000):
    all_scores = np.array(genuine_scores + impostor_scores)
    t_min, t_max = all_scores.min(), all_scores.max()
    thresholds = np.linspace(t_min, t_max, steps)

    best_far, best_frr, best_eer, best_thr = None, None, None, None
    best_gap = 1e9

    for thr in thresholds:
        far = np.mean(np.array(impostor_scores) >= thr)
        frr = np.mean(np.array(genuine_scores) < thr)
        gap = abs(far - frr)

        if gap < best_gap:
            best_gap = gap
            best_far = far
            best_frr = frr
            best_eer = (far + frr) / 2.0
            best_thr = thr

    return best_far, best_frr, best_eer, best_thr


def build_scores(embeddings, labels, max_pairs_per_class=60, seed=42):
    rng = np.random.default_rng(seed)

    genuine_scores = []
    impostor_scores = []

    unique_labels = np.unique(labels)
    class_to_idx = {c: np.where(labels == c)[0] for c in unique_labels}

    # genuine
    for c in unique_labels:
        idxs = class_to_idx[c]
        if len(idxs) < 2:
            continue

        pairs = min(max_pairs_per_class, len(idxs) * 3)
        for _ in range(pairs):
            i, j = rng.choice(idxs, size=2, replace=False)
            genuine_scores.append(cosine_similarity(embeddings[i], embeddings[j]))

    # impostor
    all_idxs = np.arange(len(labels))
    for _ in range(len(genuine_scores)):
        i = rng.choice(all_idxs)
        j = rng.choice(all_idxs)
        while labels[i] == labels[j]:
            j = rng.choice(all_idxs)
        impostor_scores.append(cosine_similarity(embeddings[i], embeddings[j]))

    return genuine_scores, impostor_scores


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc="Embedding", leave=False):
        raw = raw.to(device)
        clahe = clahe.to(device)

        t0 = time.time()
        f, g = model(raw, clahe)
        t1 = time.time()

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / raw.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    # PolyU test set
    polyu_test = HELMMSPalmVeinDataset(ROOT, "PolyU-Pure", "test", BAND, 128)
    polyu_loader = DataLoader(polyu_test, batch_size=32, shuffle=False, num_workers=0)

    # Load ECG Scalar embedder
    model = ECGScalarEmbedder().to(device)
    ckpt = torch.load("outputs/ecg_scalar_casia.pth", map_location=device)
    model.load_state_dict(ckpt, strict=False)

    embeddings, labels, avg_time = extract_embeddings(model, polyu_loader, device)
    genuine, impostor = build_scores(embeddings, labels)

    far, frr, eer, thr = compute_far_frr_eer(genuine, impostor)

    print("\n=== Cross-Dataset Evaluation ===")
    print("Train: CASIA-Pure")
    print("Test : PolyU-Pure")
    print(f"FAR: {far*100:.2f}%")
    print(f"FRR: {frr*100:.2f}%")
    print(f"EER: {eer*100:.2f}%")
    print(f"Threshold@EER: {thr:.4f}")
    print(f"Avg inference time per image: {avg_time*1000:.2f} ms")


if __name__ == "__main__":
    main()
