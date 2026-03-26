import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.dual_fusion import ResNet18FeatureExtractor
from models.dual_embedders import DualConcatEmbedder, DualSumEmbedder
from models.ecg_embedder import ECGScalarEmbedder
from models.ecg_embedder_vector import ECGVectorEmbedder


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

    # impostor (same count)
    all_idxs = np.arange(len(labels))
    for _ in range(len(genuine_scores)):
        i = rng.choice(all_idxs)
        j = rng.choice(all_idxs)
        while labels[i] == labels[j]:
            j = rng.choice(all_idxs)
        impostor_scores.append(cosine_similarity(embeddings[i], embeddings[j]))

    return genuine_scores, impostor_scores


@torch.no_grad()
def extract_single_stream(model, loader, device, mode="raw"):
    model.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc=f"Embedding({mode})", leave=False):
        x = raw if mode == "raw" else clahe
        x = x.to(device)

        t0 = time.time()
        f = model(x)
        t1 = time.time()

        f = torch.nn.functional.normalize(f, p=2, dim=1)

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / x.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


@torch.no_grad()
def extract_dual(model, loader, device, name="dual"):
    model.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc=f"Embedding({name})", leave=False):
        raw = raw.to(device)
        clahe = clahe.to(device)

        t0 = time.time()
        f = model(raw, clahe)
        t1 = time.time()

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / raw.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


@torch.no_grad()
def extract_ecg_scalar(model, loader, device):
    model.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc="Embedding(ECG-Scalar)", leave=False):
        raw = raw.to(device)
        clahe = clahe.to(device)

        t0 = time.time()
        f, g = model(raw, clahe)
        t1 = time.time()

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / raw.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


def evaluate(name, embeddings, labels, avg_time):
    genuine, impostor = build_scores(embeddings, labels)
    far, frr, eer, thr = compute_far_frr_eer(genuine, impostor)

    return {
        "Model": name,
        "FAR": far * 100,
        "FRR": frr * 100,
        "EER": eer * 100,
        "Time": avg_time * 1000,
        "Thr": thr
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    # Cross dataset test: PolyU-Pure
    test_ds = HELMMSPalmVeinDataset(ROOT, "PolyU-Pure", "test", BAND, 128)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    results = []

    # RAW
    raw_model = ResNet18FeatureExtractor().to(device)
    raw_ckpt = torch.load("outputs/raw_resnet18_casia.pth", map_location=device)
    raw_model.load_state_dict(raw_ckpt, strict=False)

    emb, lab, t = extract_single_stream(raw_model, test_loader, device, mode="raw")
    results.append(evaluate("Single-Stream (RAW)", emb, lab, t))

    # CLAHE
    clahe_model = ResNet18FeatureExtractor().to(device)
    clahe_ckpt = torch.load("outputs/clahe_resnet18_casia.pth", map_location=device)
    clahe_model.load_state_dict(clahe_ckpt, strict=False)

    emb, lab, t = extract_single_stream(clahe_model, test_loader, device, mode="clahe")
    results.append(evaluate("Single-Stream (CLAHE)", emb, lab, t))

    # Dual CONCAT
    concat_model = DualConcatEmbedder().to(device)
    concat_ckpt = torch.load("outputs/dual_concat_casia.pth", map_location=device)
    concat_model.load_state_dict(concat_ckpt, strict=False)

    emb, lab, t = extract_dual(concat_model, test_loader, device, name="Dual-Concat")
    results.append(evaluate("Dual-Stream (CONCAT)", emb, lab, t))

    # Dual SUM
    sum_model = DualSumEmbedder().to(device)
    sum_ckpt = torch.load("outputs/dual_sum_casia.pth", map_location=device)
    sum_model.load_state_dict(sum_ckpt, strict=False)

    emb, lab, t = extract_dual(sum_model, test_loader, device, name="Dual-Sum")
    results.append(evaluate("Dual-Stream (SUM)", emb, lab, t))

    # ECG Scalar
    ecg_scalar = ECGScalarEmbedder().to(device)
    ecg_ckpt = torch.load("outputs/ecg_scalar_casia.pth", map_location=device)
    ecg_scalar.load_state_dict(ecg_ckpt, strict=False)

    emb, lab, t = extract_ecg_scalar(ecg_scalar, test_loader, device)
    results.append(evaluate("Proposed ECG (Scalar)", emb, lab, t))

    # ECG Vector
    ecg_vector = ECGVectorEmbedder().to(device)
    ecg_vec_ckpt = torch.load("outputs/ecg_vector_casia.pth", map_location=device)
    ecg_vector.load_state_dict(ecg_vec_ckpt, strict=False)

    emb, lab, t = extract_dual(ecg_vector, test_loader, device, name="ECG-Vector")
    results.append(evaluate("Proposed ECG (Vector)", emb, lab, t))

    print("\n================ CROSS DATASET TABLE ================\n")
    print("Train: CASIA-Pure | Test: PolyU-Pure\n")

    for r in results:
        print(
            f"{r['Model']:<26} | "
            f"FAR: {r['FAR']:.2f}% | "
            f"FRR: {r['FRR']:.2f}% | "
            f"EER: {r['EER']:.2f}% | "
            f"Time: {r['Time']:.2f} ms"
        )


if __name__ == "__main__":
    main()
