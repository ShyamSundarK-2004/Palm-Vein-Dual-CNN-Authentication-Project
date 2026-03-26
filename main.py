<<<<<<< HEAD
# main.py
import torch
from models.dual_channel_model import DualChannelPalmVeinCNN
from evaluation.result_formatter import print_authentication_result
from explainability.grad_cam import GradCAM

# -------------------------------
# Experiment Modes
# -------------------------------
MODE_RAW = "raw"
MODE_CLAHE = "clahe"
MODE_DUAL = "dual"


def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔁 SELECT EXPERIMENT MODE HERE
        #EXPERIMENT_MODE = MODE_DUAL
        EXPERIMENT_MODE = MODE_RAW
        # EXPERIMENT_MODE = MODE_CLAHE

        ENABLE_GRADCAM = True

        # -------------------------------
        # Model Initialization
        # -------------------------------
        model = DualChannelPalmVeinCNN(num_classes=100)
        model.to(device)
        model.eval()

        # -------------------------------
        # Dummy Inputs (Real dataset later)
        # -------------------------------
        raw = torch.randn(1, 1, 128, 128, device=device, requires_grad=True)
        enhanced = torch.randn(1, 1, 128, 128, device=device, requires_grad=True)

        # -------------------------------
        # Grad-CAM SETUP (MUST be BEFORE forward)
        # -------------------------------
        if ENABLE_GRADCAM:
            if EXPERIMENT_MODE == MODE_RAW:
                target_layer = model.raw_branch.features[6]      # last Conv2d
                mode_name = "SINGLE-CHANNEL (RAW)"

            elif EXPERIMENT_MODE == MODE_CLAHE:
                target_layer = model.enhanced_branch.features[6]
                mode_name = "SINGLE-CHANNEL (CLAHE)"

            elif EXPERIMENT_MODE == MODE_DUAL:
                target_layer = model.raw_branch.features[6]
                mode_name = "DUAL-CHANNEL (RAW + CLAHE)"

            else:
                raise ValueError("Invalid EXPERIMENT_MODE")

            grad_cam = GradCAM(model, target_layer)

        # -------------------------------
        # Forward Pass
        # -------------------------------
        if EXPERIMENT_MODE == MODE_RAW:
            logits = model(raw, raw)

        elif EXPERIMENT_MODE == MODE_CLAHE:
            logits = model(enhanced, enhanced)

        elif EXPERIMENT_MODE == MODE_DUAL:
            logits = model(raw, enhanced)

        else:
            raise ValueError("Invalid EXPERIMENT_MODE")

        # -------------------------------
        # Backward Pass (Grad-CAM trigger)
        # -------------------------------
        if ENABLE_GRADCAM:
            pred_class = logits.argmax(dim=1)
            score = logits[0, pred_class]

            model.zero_grad()
            score.backward()

            cam = grad_cam.generate(score)
            print("✅ Grad-CAM generated successfully (structure check only)")

        # -------------------------------
        # Decision Logic (Dataset-dependent)
        # -------------------------------
        decision = "PENDING (Dataset Not Loaded)"

        # -------------------------------
        # Metrics (Placeholders now)
        # -------------------------------
        metrics = {
            "accuracy": "N/A",
            "far": "N/A",
            "frr": "N/A",
            "eer": "N/A"
        }

        # -------------------------------
        # Inference Time (Placeholder)
        # -------------------------------
        inference_time = "N/A (Dataset Not Loaded)"

        # -------------------------------
        # Output (IEEE-style)
        # -------------------------------
        print_authentication_result(
            decision=decision,
            metrics=metrics,
            inference_time=inference_time,
            mode=mode_name
        )

    except Exception as e:
        print("❌ Execution failed:", str(e))


if __name__ == "__main__":
    main()
=======
# main.py
import torch
from models.dual_channel_model import DualChannelPalmVeinCNN
from evaluation.result_formatter import print_authentication_result
from explainability.grad_cam import GradCAM

# -------------------------------
# Experiment Modes
# -------------------------------
MODE_RAW = "raw"
MODE_CLAHE = "clahe"
MODE_DUAL = "dual"


def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔁 SELECT EXPERIMENT MODE HERE
        #EXPERIMENT_MODE = MODE_DUAL
        EXPERIMENT_MODE = MODE_RAW
        # EXPERIMENT_MODE = MODE_CLAHE

        ENABLE_GRADCAM = True

        # -------------------------------
        # Model Initialization
        # -------------------------------
        model = DualChannelPalmVeinCNN(num_classes=100)
        model.to(device)
        model.eval()

        # -------------------------------
        # Dummy Inputs (Real dataset later)
        # -------------------------------
        raw = torch.randn(1, 1, 128, 128, device=device, requires_grad=True)
        enhanced = torch.randn(1, 1, 128, 128, device=device, requires_grad=True)

        # -------------------------------
        # Grad-CAM SETUP (MUST be BEFORE forward)
        # -------------------------------
        if ENABLE_GRADCAM:
            if EXPERIMENT_MODE == MODE_RAW:
                target_layer = model.raw_branch.features[6]      # last Conv2d
                mode_name = "SINGLE-CHANNEL (RAW)"

            elif EXPERIMENT_MODE == MODE_CLAHE:
                target_layer = model.enhanced_branch.features[6]
                mode_name = "SINGLE-CHANNEL (CLAHE)"

            elif EXPERIMENT_MODE == MODE_DUAL:
                target_layer = model.raw_branch.features[6]
                mode_name = "DUAL-CHANNEL (RAW + CLAHE)"

            else:
                raise ValueError("Invalid EXPERIMENT_MODE")

            grad_cam = GradCAM(model, target_layer)

        # -------------------------------
        # Forward Pass
        # -------------------------------
        if EXPERIMENT_MODE == MODE_RAW:
            logits = model(raw, raw)

        elif EXPERIMENT_MODE == MODE_CLAHE:
            logits = model(enhanced, enhanced)

        elif EXPERIMENT_MODE == MODE_DUAL:
            logits = model(raw, enhanced)

        else:
            raise ValueError("Invalid EXPERIMENT_MODE")

        # -------------------------------
        # Backward Pass (Grad-CAM trigger)
        # -------------------------------
        if ENABLE_GRADCAM:
            pred_class = logits.argmax(dim=1)
            score = logits[0, pred_class]

            model.zero_grad()
            score.backward()

            cam = grad_cam.generate(score)
            print("✅ Grad-CAM generated successfully (structure check only)")

        # -------------------------------
        # Decision Logic (Dataset-dependent)
        # -------------------------------
        decision = "PENDING (Dataset Not Loaded)"

        # -------------------------------
        # Metrics (Placeholders now)
        # -------------------------------
        metrics = {
            "accuracy": "N/A",
            "far": "N/A",
            "frr": "N/A",
            "eer": "N/A"
        }

        # -------------------------------
        # Inference Time (Placeholder)
        # -------------------------------
        inference_time = "N/A (Dataset Not Loaded)"

        # -------------------------------
        # Output (IEEE-style)
        # -------------------------------
        print_authentication_result(
            decision=decision,
            metrics=metrics,
            inference_time=inference_time,
            mode=mode_name
        )

    except Exception as e:
        print("❌ Execution failed:", str(e))


if __name__ == "__main__":
    main()
>>>>>>> 793e70482332d6cabced4d75d1a53ed9a9c1a2f6
