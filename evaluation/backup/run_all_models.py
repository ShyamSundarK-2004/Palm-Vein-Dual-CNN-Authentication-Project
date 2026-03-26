import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.dual_fusion import ResNet18FeatureExtractor
from models.ecg_embedder import ECGScalarEmbedder
from models.ecg_embedder_vector import ECGVectorEmbedder
from models.dual_embedders import DualConcatEmbedder, DualSumEmbedder


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
def extract_embeddings_single_stream(feature_model, loader, device, mode="raw"):
    feature_model.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc=f"Embedding({mode})", leave=False):
        x = raw if mode == "raw" else clahe
        x = x.to(device)

        t0 = time.time()
        f = feature_model(x)
        t1 = time.time()

        f = torch.nn.functional.normalize(f, p=2, dim=1)

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / x.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


@torch.no_grad()
def extract_embeddings_dual(embedder, loader, device, name="dual"):
    embedder.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc=f"Embedding({name})", leave=False):
        raw = raw.to(device)
        clahe = clahe.to(device)

        t0 = time.time()
        f = embedder(raw, clahe)
        t1 = time.time()

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / raw.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


@torch.no_grad()
def extract_embeddings_ecg_scalar(embedder, loader, device):
    embedder.eval()
    embs, labs, times = [], [], []

    for raw, clahe, y in tqdm(loader, desc="Embedding(ECG-Scalar)", leave=False):
        raw = raw.to(device)
        clahe = clahe.to(device)

        t0 = time.time()
        f, g = embedder(raw, clahe)
        t1 = time.time()

        embs.append(f.cpu().numpy())
        labs.append(y.numpy())
        times.append((t1 - t0) / raw.size(0))

    return np.vstack(embs), np.concatenate(labs), float(np.mean(times))


def evaluate_model(name, embeddings, labels, avg_time):
    genuine, impostor = build_scores(embeddings, labels)
    far, frr, eer, thr = compute_far_frr_eer(genuine, impostor)

    return {
        "Model": name,
        "FAR(%)": far * 100,
        "FRR(%)": frr * 100,
        "EER(%)": eer * 100,
        "Time(ms)": avg_time * 1000
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    test_ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "test", BAND, 128)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    results = []

    # 1) RAW
    raw_extractor = ResNet18FeatureExtractor().to(device)
    raw_ckpt = torch.load("outputs/raw_resnet18_casia.pth", map_location=device)
    raw_extractor.load_state_dict(raw_ckpt, strict=False)

    emb, lab, t = extract_embeddings_single_stream(raw_extractor, test_loader, device, mode="raw")
    results.append(evaluate_model("Single-Stream (RAW)", emb, lab, t))

    # 2) CLAHE
    clahe_extractor = ResNet18FeatureExtractor().to(device)
    clahe_ckpt = torch.load("outputs/clahe_resnet18_casia.pth", map_location=device)
    clahe_extractor.load_state_dict(clahe_ckpt, strict=False)

    emb, lab, t = extract_embeddings_single_stream(clahe_extractor, test_loader, device, mode="clahe")
    results.append(evaluate_model("Single-Stream (CLAHE)", emb, lab, t))

    # 3) Dual CONCAT
    dual_concat = DualConcatEmbedder().to(device)
    concat_ckpt = torch.load("outputs/dual_concat_casia.pth", map_location=device)
    dual_concat.load_state_dict(concat_ckpt, strict=False)

    emb, lab, t = extract_embeddings_dual(dual_concat, test_loader, device, name="Dual-Concat")
    results.append(evaluate_model("Dual-Stream (CONCAT)", emb, lab, t))

    # 4) Dual SUM
    dual_sum = DualSumEmbedder().to(device)
    sum_ckpt = torch.load("outputs/dual_sum_casia.pth", map_location=device)
    dual_sum.load_state_dict(sum_ckpt, strict=False)

    emb, lab, t = extract_embeddings_dual(dual_sum, test_loader, device, name="Dual-Sum")
    results.append(evaluate_model("Dual-Stream (SUM)", emb, lab, t))

    # 5) ECG Scalar
    ecg_scalar = ECGScalarEmbedder().to(device)
    ecg_scalar_ckpt = torch.load("outputs/ecg_scalar_casia.pth", map_location=device)
    ecg_scalar.load_state_dict(ecg_scalar_ckpt, strict=False)

    emb, lab, t = extract_embeddings_ecg_scalar(ecg_scalar, test_loader, device)
    results.append(evaluate_model("Proposed ECG (Scalar)", emb, lab, t))

    # 6) ECG Vector
    ecg_vector = ECGVectorEmbedder().to(device)
    ecg_vector_ckpt = torch.load("outputs/ecg_vector_casia.pth", map_location=device)
    ecg_vector.load_state_dict(ecg_vector_ckpt, strict=False)

    emb, lab, t = extract_embeddings_dual(ecg_vector, test_loader, device, name="ECG-Vector")
    results.append(evaluate_model("Proposed ECG (Vector)", emb, lab, t))

    print("\n================ FINAL TABLE (CASIA-Pure TEST) ================\n")
    for r in results:
        print(
            f"{r['Model']:<26} | "
            f"FAR: {r['FAR(%)']:.2f}% | "
            f"FRR: {r['FRR(%)']:.2f}% | "
            f"EER: {r['EER(%)']:.2f}% | "
            f"Time: {r['Time(ms)']:.2f} ms"
        )


if __name__ == "__main__":
    main()
