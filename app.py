import streamlit as st
import torch
import cv2
import numpy as np
import time
from PIL import Image
import torchvision.transforms as transforms

from models.raw_cnn import RawResNet18
from models.dual_fusion import ConcatFusionModel
from models.ecg_model import ECGFusionModel

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- CONFIG ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="Palm Vein Authentication", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #020617, #0f172a); color:white;}
.title {font-size:45px; font-weight:bold; color:#00ffc6; text-align:center; padding:20px;}
</style>
""", unsafe_allow_html=True)

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    transforms.Normalize([0.5], [0.5])
])

# ---------------- MODELS ----------------
@st.cache_resource
def load_models():
    try:
        casia_raw = RawResNet18(100)
        casia_raw.load_state_dict(torch.load("outputs/CASIA-Pure_raw.pth", map_location=device))

        casia_concat = ConcatFusionModel(100)
        casia_concat.load_state_dict(torch.load("outputs/CASIA-Pure_concat.pth", map_location=device))

        casia_ecg = ECGFusionModel(100)
        casia_ecg.load_state_dict(torch.load("outputs/CASIA-Pure_ecg.pth", map_location=device), strict=False)

        polyu_ecg = ECGFusionModel(500)
        polyu_ecg.load_state_dict(torch.load("outputs/PolyU-Pure_ecg.pth", map_location=device), strict=False)

        return (
            casia_raw.to(device).eval(),
            casia_concat.to(device).eval(),
            casia_ecg.to(device).eval(),
            polyu_ecg.to(device).eval()
        )

    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None, None, None, None


casia_raw, casia_concat, casia_ecg, polyu_ecg = load_models()

model_map = {}
if casia_raw is not None:
    model_map = {
        "CASIA - RAW": (casia_raw, "raw"),
        "CASIA - CONCAT": (casia_concat, "dual"),
        "CASIA - ECG": (casia_ecg, "dual"),
        "POLYU - ECG": (polyu_ecg, "dual")
    }

# ---------------- HELPERS ----------------
def process_image(file):
    img = Image.open(file).convert("L")
    img_np = np.array(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_np)

    raw_tensor = transform(img_np).unsqueeze(0).to(device)
    enh_tensor = transform(clahe_img).unsqueeze(0).to(device)

    return img, raw_tensor, enh_tensor


def predict(model, raw, enh, mode):
    start = time.time()
    with torch.no_grad():
        out = model(raw) if mode == "raw" else model(raw, enh)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), (time.time() - start) * 1000


# 🔥 GRADCAM WRAPPER (ONLY RAW INPUT)
class GradCAMWrapper(torch.nn.Module):
    def __init__(self, model, mode, enh):
        super().__init__()
        self.model = model
        self.mode = mode
        self.enh = enh

    def forward(self, x):
        if self.mode == "raw":
            return self.model(x)
        else:
            out = self.model(x, self.enh)
            if isinstance(out, tuple):
                out = out[0]
            return out
def find_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    return None

def get_gradcam(model, raw, enh, model_type):

    try:
        wrapper = GradCAMWrapper(model, model_type, enh)

        # 🔥 AUTO FIND CORRECT LAYER
        target_layer = find_last_conv_layer(model)

        if target_layer is None:
            print("No conv layer found")
            return None

        cam = GradCAM(model=wrapper, target_layers=[target_layer])

        # 🔥 ONLY FIRST INPUT (RAW)
        grayscale_cam = cam(input_tensor=raw)[0]

        # 🔥 ORIGINAL IMAGE
        img = raw.squeeze().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_rgb = np.stack([img]*3, axis=-1)

        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        return visualization

    except Exception as e:
        print("GradCAM Error:", e)
        return None

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["CASIA - RAW", "CASIA - CONCAT", "CASIA - ECG", "POLYU - ECG"]
)

st.sidebar.markdown("### 📂 Menu")

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if st.sidebar.button("📊 Overview"):
    st.session_state.page = "Overview"

if st.sidebar.button("🚀 Demo"):
    st.session_state.page = "Demo"

if st.sidebar.button("ℹ️ About"):
    st.session_state.page = "About"

page = st.session_state.page

# ---------------- TITLE ----------------
st.markdown('<div class="title">🌴 Palm Vein Authentication System</div>', unsafe_allow_html=True)

# ---------------- OVERVIEW ----------------
if page == "Overview":

    st.markdown("## 📌 System Overview")

    st.markdown("""
### 🌴 Palm Vein Authentication System

A deep learning-based biometric system that identifies individuals using subcutaneous vascular patterns.

---

### 🧠 Architecture
- Dual-channel CNN (RAW + CLAHE)
- Feature Fusion (Concat / ECG)
- Enhancement-aware learning

---

### ⚡ Key Innovation
- Enhancement Confidence Gate (ECG)
- Dynamically controls enhancement contribution

---

### 🎯 Objective
- Improve robustness under noisy conditions
- Handle cross-dataset variation
- Reduce FAR / FRR / EER

---

### 🔥 Explainability
- Grad-CAM highlights vein regions
- Helps understand model decisions

---

### 📊 Models Supported
- CASIA - RAW
- CASIA - CONCAT
- CASIA - ECG
- POLYU - ECG
""")

# ---------------- ABOUT ----------------
elif page == "About":

    st.markdown("## ℹ️ About This Project")

    st.markdown("""
### 📄 Enhancement-Aware Dual-Channel CNN

This project proposes a palm vein authentication system using deep learning.

---

### 💡 Core Idea
- Use RAW image  
- Use CLAHE enhanced image  
- Learn when enhancement is useful  

---

### ⚙️ ECG Mechanism
- Learns reliability of enhancement  
- Suppresses noise  
- Improves robustness  

---

### 🧪 Datasets
- CASIA-Pure  
- PolyU-Pure  

---

### 📈 Results
- Improved Accuracy  
- Reduced Error Rates  
- Better Generalization  

---

### 🔥 Explainability
Grad-CAM shows model focus on vein regions.
""")

# ---------------- DEMO ----------------
elif page == "Demo":

    if not model_map:
        st.error("Models not loaded.")
    else:
        model, model_type = model_map[model_choice]

        tab1, tab2 = st.tabs(["🆔 Identification", "🔐 Authentication"])

        # -------- IDENTIFICATION --------
        with tab1:
            file = st.file_uploader("Upload Palm", type=["jpg", "png"])

            if file:
                img, raw, enh = process_image(file)

                col1, col2 = st.columns(2)
                col1.image(img)

                pred_id, confidence, inf_time = predict(model, raw, enh, model_type)

                with col2:
                    st.success(f"User ID: {pred_id}")
                    st.metric("Confidence", f"{confidence:.4f}")
                    st.metric("Latency", f"{inf_time:.2f} ms")

        # -------- AUTHENTICATION --------
        with tab2:
            f1 = st.file_uploader("Image 1", key="auth1")
            f2 = st.file_uploader("Image 2", key="auth2")

            if f1 and f2:
                _, r1, e1 = process_image(f1)
                _, r2, e2 = process_image(f2)

                p1, _, _ = predict(model, r1, e1, model_type)
                p2, _, _ = predict(model, r2, e2, model_type)

                if p1 == p2:
                    st.success(f"✅ MATCH: ID {p1}")
                else:
                    st.error("❌ NO MATCH")

                st.info("Model focuses on highlighted vein regions")

                st.markdown("### 🔥 Why this decision? (Grad-CAM)")

                cam1 = get_gradcam(model, r1, e1, model_type)
                cam2 = get_gradcam(model, r2, e2, model_type)

                colA, colB = st.columns(2)

                if cam1 is not None:
                    colA.image(cam1, caption="Image 1 Attention")

                if cam2 is not None:
                    colB.image(cam2, caption="Image 2 Attention")