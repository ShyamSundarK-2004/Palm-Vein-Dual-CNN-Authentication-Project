import os
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from datasets.helmms_dataset import HELMMSPalmVeinDataset
from models.ecg_fusion_gradcam import ECGFusionScalarGradCAM


def tensor_to_rgb01(x):
    img = x.squeeze(0).cpu().numpy()
    rgb = np.stack([img, img, img], axis=-1)
    return rgb.astype(np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    os.makedirs("outputs/gradcam", exist_ok=True)

    ROOT = r"data/raw/HELM-MS"
    BAND = "850"

    ds = HELMMSPalmVeinDataset(ROOT, "CASIA-Pure", "test", BAND, 128)

    model = ECGFusionScalarGradCAM(num_classes=ds.num_classes).to(device)

    ckpt = torch.load("outputs/ecg_scalar_casia.pth", map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    # Target layer: last conv in ResNet18
    raw_target_layer = model.raw_backbone.layer4[-1].conv2
    enh_target_layer = model.enh_backbone.layer4[-1].conv2

    cam_raw = GradCAM(model=model.raw_backbone, target_layers=[raw_target_layer])
    cam_enh = GradCAM(model=model.enh_backbone, target_layers=[enh_target_layer])

    sample_indices = [0, 5, 10, 25, 40]

    for idx in sample_indices:
        raw, clahe, label = ds[idx]
        raw_in = raw.unsqueeze(0).to(device)
        clahe_in = clahe.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, gate = model(raw_in, clahe_in)
            pred = logits.argmax(dim=1).item()
            gate_val = float(gate.item())

        # RAW CAM
        raw_cam = cam_raw(input_tensor=raw_in)[0]
        raw_rgb = tensor_to_rgb01(raw)
        raw_vis = show_cam_on_image(raw_rgb, raw_cam, use_rgb=True)

        # CLAHE CAM
        enh_cam = cam_enh(input_tensor=clahe_in)[0]
        enh_rgb = tensor_to_rgb01(clahe)
        enh_vis = show_cam_on_image(enh_rgb, enh_cam, use_rgb=True)

        raw_path = f"outputs/gradcam/sample_{idx}_RAW_pred{pred}_gate{gate_val:.2f}.png"
        enh_path = f"outputs/gradcam/sample_{idx}_CLAHE_pred{pred}_gate{gate_val:.2f}.png"

        cv2.imwrite(raw_path, cv2.cvtColor(raw_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(enh_path, cv2.cvtColor(enh_vis, cv2.COLOR_RGB2BGR))

        print("Saved:", raw_path)
        print("Saved:", enh_path)

    print("\nDone. Check outputs/gradcam/")

def generate_gradcam(model, raw, clahe, device):

    raw_in = raw.unsqueeze(0).to(device)
    clahe_in = clahe.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, gate = model(raw_in, clahe_in)
        pred = logits.argmax(dim=1).item()
        gate_val = float(gate.item())

    # GradCAM objects
    raw_target = model.raw_backbone.layer4[-1].conv2
    enh_target = model.enh_backbone.layer4[-1].conv2

    cam_raw = GradCAM(model=model.raw_backbone, target_layers=[raw_target])
    cam_enh = GradCAM(model=model.enh_backbone, target_layers=[enh_target])

    raw_cam = cam_raw(input_tensor=raw_in)[0]
    enh_cam = cam_enh(input_tensor=clahe_in)[0]

    raw_rgb = tensor_to_rgb01(raw)
    enh_rgb = tensor_to_rgb01(clahe)

    raw_vis = show_cam_on_image(raw_rgb, raw_cam, use_rgb=True)
    enh_vis = show_cam_on_image(enh_rgb, enh_cam, use_rgb=True)

    return raw_vis, enh_vis, pred, gate_val


if __name__ == "__main__":
    main()
