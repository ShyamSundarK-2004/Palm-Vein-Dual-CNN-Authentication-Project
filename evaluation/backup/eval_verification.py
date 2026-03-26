import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.ecg_embedder import ECGScalarEmbedder


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def compute_far_frr_eer(genuine_scores, impostor_scores, steps=1000):
    """
    Sweep threshold from min to max and compute FAR/FRR.
    Return EER and threshold at EER.
    """
    all_scores = np.array(genuine_scores + impostor_scores)
    t_min, t_max = all_scores.min(), all_scores.max()

    thresholds = np.linspace(t_min, t_max, steps)

    best_eer = 1.0
    best_thr = None
    best_far = None
    best_frr = None

    for thr in thresholds:
        far = np.mean(np.array(impostor_scores) >= thr)  # impostor accepted
        frr = np.mean(np.array(genuine_scores) < thr)    # genuine rejected

        eer = (far + frr) / 2.0
        if abs(far - frr) < abs(best_far - best_frr) if best_far is not None else True:
            best_eer = eer
            best_thr = thr
            best_far = far
            best_frr = frr

    return best_far, best_frr, best_eer, best_thr


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    times = []

    for raw, clahe, y in tqdm(loader, desc="Embedding", leave=False):
        raw = raw.to(device)
        clahe = clahe.to(device)

        t0 = time.time()
        emb, gate = model(raw, clahe)
        t1 = time.time()

        embeddings.append(emb.cpu().numpy())
        labels.append(y.numpy())
        times.append((t1 - t0) / raw.size(0))

    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    avg_time = float(np.mean(times))

    return embeddings, labels, avg_time


def build_scores(embeddings, labels, max_pairs_per_class=50, seed=42):
    """
    Build genuine and impostor cosine similarity scores.
    """
    rng = np.random.default_rng(seed)

    genuine_scores = []
    impostor_scores = []

    unique_labels = np.unique(labels)

    # indices per class
    class_to_idx = {c: np.where(labels == c)[0] for c in unique_labels}

    # genuine pairs
    for c in unique_labels:
        idxs = class_to_idx[c]
        if len(idxs) < 2:
            continue

        # sample some pairs
        for _ in range(min(max_pairs_per_class, len(idxs) * 2)):
            i, j = rng.choice(idxs, size=2, replace=False)
            s = cosine_similarity(embeddings[i], embeddings[j])
            genuine_scores.append(s)

    # impostor pairs
    all_idxs = np.arange(len(labels))
    for _ in range(len(genuine_scores)):
        i = rng.choice(all_idxs)
        j = rng.choice(all_idxs)
        while labels[i] == labels[j]:
            j = rng.choice(all_idxs)

        s = cosine_similarity(embeddings[i], embeddings[j])
        impostor_scores.append(s)

    return genuine_scores, impostor_scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    # Use TEST split for verification scoring
    test_ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "test", BAND, 128)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load embedder
    model = ECGScalarEmbedder().to(device)

    # Load weights from your trained scalar ECG classifier
    # NOTE: classifier weights will be ignored automatically since embedder has no classifier
    ckpt = torch.load("outputs/ecg_scalar_casia.pth", map_location=device)
    model.load_state_dict(ckpt, strict=False)

    embeddings, labels, avg_time = extract_embeddings(model, test_loader, device)

    genuine_scores, impostor_scores = build_scores(embeddings, labels)

    far, frr, eer, thr = compute_far_frr_eer(genuine_scores, impostor_scores)

    print("\n=== Verification Results (CASIA-Pure TEST) ===")
    print(f"Genuine pairs:  {len(genuine_scores)}")
    print(f"Impostor pairs: {len(impostor_scores)}")
    print(f"FAR: {far*100:.2f}%")
    print(f"FRR: {frr*100:.2f}%")
    print(f"EER: {eer*100:.2f}%")
    print(f"Threshold@EER: {thr:.4f}")
    print(f"Avg inference time per image: {avg_time*1000:.2f} ms")


if __name__ == "__main__":
    main()
