import os
import glob
from typing import List, Tuple, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def parse_helmms_filename(fname):
    base = os.path.basename(fname)
    name = os.path.splitext(base)[0]
    parts = name.split("_")

    if len(parts) >= 4:  # CASIA
        return {
            "subject_id": parts[0],
            "band": parts[2]
        }

    if len(parts) == 3:  # PolyU
        return {
            "subject_id": parts[1],
            "band": "850"
        }

    raise ValueError(f"Unexpected filename: {base}")


def get_image_paths(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    return sorted(paths)


class HELMMSPalmVeinDataset(Dataset):

    def __init__(self, root_dir, dataset_name="CASIA-Pure",
                 band="850", img_size=128, transform=None):

        self.transform = transform
        self.img_size = img_size

        dataset_path = os.path.join(root_dir, dataset_name)

        all_paths = get_image_paths(dataset_path)

        self.samples = []
        subjects = []

        for p in all_paths:
            meta = parse_helmms_filename(p)

            # filter only for CASIA
            if dataset_name == "CASIA-Pure" and meta["band"] != band:
                continue

            self.samples.append(p)
            subjects.append(meta["subject_id"])

        # labels
        unique_subjects = sorted(set(subjects))
        self.subject_to_label = {sid: i for i, sid in enumerate(unique_subjects)}
        self.num_classes = len(unique_subjects)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path = self.samples[idx]
        meta = parse_helmms_filename(path)

        label = self.subject_to_label[meta["subject_id"]]

        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (self.img_size, self.img_size))

        clahe = apply_clahe(gray)

        if self.transform:
            raw = self.transform(gray)
            clahe = self.transform(clahe)
        else:
            gray = gray.astype(np.float32) / 255.0
            clahe = clahe.astype(np.float32) / 255.0
            raw = torch.from_numpy(gray).unsqueeze(0)
            clahe = torch.from_numpy(clahe).unsqueeze(0)

        return raw, clahe, label