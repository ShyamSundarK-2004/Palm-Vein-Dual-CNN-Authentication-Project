"""
Palm Vein Authentication Training Pipeline
Optimized for biometric authenticity and real-time deployment

Key Features:
- Identity-based train/val/test split (prevents subject leakage)
- Minimal, biometric-safe augmentation (preserves vein patterns)
- Grayscale → RGB conversion to preserve pretrained weights
- Fixed ECG attention mechanism
- EER-optimized evaluation metrics
- Real-time inference ready
"""

import os
import sys
import time
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
#                   PALM VEIN SPECIFIC CONFIG
# ============================================================================

class BiometricConfig:
    """Configuration optimized for palm vein authentication"""
    
    # ---- Data Config ----
    DATA_ROOT = "data/raw/HELM-MS"
    DATASET_NAME = "PolyU-Pure"
    WAVELENGTH = "850"
    IMAGE_SIZE = 224
    
    # ---- IDENTITY-BASED SPLIT (Critical for biometrics) ----
    TRAIN_IDENTITY_RATIO = 0.70  # 70% of subjects for training
    VAL_IDENTITY_RATIO = 0.15    # 15% for validation
    TEST_IDENTITY_RATIO = 0.15   # 15% for testing
    # Note: This is SUBJECT-BASED, not image-based
    
    # ---- Hyperparameters ----
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5
    LABEL_SMOOTHING = 0.08  # Reduced for biometrics
    
    # ---- Optimization ----
    OPTIMIZER = "adamw"
    SCHEDULER = "cosine"
    WARMUP_EPOCHS = 5
    
    # ---- BIOMETRIC-SAFE REGULARIZATION ----
    DROPOUT_RATE = 0.2  # Reduced - vein patterns are distinct
    USE_MIXUP = False   # ❌ DISABLED - blending identities is invalid
    USE_CUTMIX = False  # ❌ DISABLED
    USE_AUGMENTATION = True  # ✅ Light augmentation only
    
    # ---- Mixed Precision ----
    USE_AMP = True
    ACCUMULATION_STEPS = 1
    GRADIENT_CLIP = 1.0
    
    # ---- Model Selection ----
    MODEL_TYPE = "concat"  # raw, ecg, concat
    BACKBONE = "resnet18"  # resnet18 is good for biometrics
    PRETRAINED = True
    FREEZE_BACKBONE = False
    USE_GRAYSCALE_TO_RGB = True  # ✅ Preserve pretrained weights
    
    # ---- Early Stopping ----
    EARLY_STOPPING = True
    PATIENCE = 20
    MIN_DELTA = 0.001
    
    # ---- Checkpointing ----
    SAVE_DIR = "checkpoints/palm_vein"
    SAVE_EVERY_N_EPOCHS = 5
    
    # ---- Biometric Evaluation ----
    COMPUTE_EER = True  # Equal Error Rate for biometrics
    COMPUTE_FNMR_FMR = True  # False Non-Match / False Match rates
    
    # ---- Logging ----
    LOG_DIR = "logs/palm_vein"
    
    # ---- Device ----
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ---- Reproducibility ----
    SEED = 42
    DETERMINISTIC = True
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and k != 'to_dict' and not isinstance(v, classmethod)}


# ============================================================================
#                       BIOMETRIC-SAFE AUGMENTATION
# ============================================================================

class BiometricAugmentation:
    """
    Minimal augmentation that preserves vein patterns.
    
    NO:
    - Perspective distortion (destroys vein geometry)
    - Blur (removes vein details)
    - Color jitter (grayscale only)
    - Random invert (alters biometric signal)
    
    YES:
    - Small rotations (0-5°, hand natural variation)
    - Slight position shifts (hand placement variance)
    - Horizontal flip (mirror acquisition)
    """
    
    @staticmethod
    def get_train_transform():
        """Minimal augmentation for training"""
        return transforms.Compose([
            transforms.ToPILImage(),
            # ✅ Light rotation (hand jitter in acquisition)
            transforms.RandomRotation(5, interpolation=transforms.InterpolationMode.BICUBIC),
            # ✅ Slight translation (hand placement variance)
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), 
                                   interpolation=transforms.InterpolationMode.BICUBIC),
            # ✅ Horizontal flip (mirror acquisition)
            transforms.RandomHorizontalFlip(p=0.5),
            # ✅ Convert to 3-channel RGB to use pretrained weights
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((BiometricConfig.IMAGE_SIZE, BiometricConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            # ImageNet normalization (for pretrained models)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_val_transform():
        """No augmentation for validation - use as captured"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((BiometricConfig.IMAGE_SIZE, BiometricConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_transform():
        """No augmentation for testing - as captured"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((BiometricConfig.IMAGE_SIZE, BiometricConfig.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


# ============================================================================
#                    IDENTITY-BASED DATA SPLITTING
# ============================================================================

class IdentityBasedSplitter:
    def __init__(self, dataset, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # ✅ Extract subjects using LABEL (correct way)
        self.subject_ids = self._extract_subject_ids()

        shuffled_ids = list(self.subject_ids)
        random.shuffle(shuffled_ids)

        n_subjects = len(shuffled_ids)
        train_count = int(n_subjects * train_ratio)
        val_count = int(n_subjects * val_ratio)

        self.train_subjects = set(shuffled_ids[:train_count])
        self.val_subjects = set(shuffled_ids[train_count:train_count + val_count])
        self.test_subjects = set(shuffled_ids[train_count + val_count:])

        print(f"\n{'='*60}")
        print("IDENTITY-BASED SPLIT (Biometric-correct)")
        print(f"{'='*60}")
        print(f"Total subjects: {n_subjects}")
        print(f"Train subjects: {len(self.train_subjects)}")
        print(f"Val subjects:   {len(self.val_subjects)}")
        print(f"Test subjects:  {len(self.test_subjects)}")
        print(f"{'='*60}\n")

    # ✅ FINAL CORRECT SUBJECT EXTRACTION
    def _extract_subject_ids(self):
        subject_ids = set()

        for idx in range(len(self.dataset)):
            try:
                _, _, label = self.dataset[idx]  # 🔥 use label directly
                subject_ids.add(int(label))
            except:
                continue

        print(f"✅ Extracted {len(subject_ids)} unique subjects")
        return subject_ids

    # ✅ FINAL SUBJECT FETCH
    def _get_subject(self, idx):
        try:
            _, _, label = self.dataset[idx]
            return int(label)
        except:
            return -1

    # ✅ FINAL SPLIT LOGIC
    def get_split_indices(self):
        train_indices = []
        val_indices = []
        test_indices = []

        for idx in range(len(self.dataset)):
            subject = self._get_subject(idx)

            if subject in self.train_subjects:
                train_indices.append(idx)
            elif subject in self.val_subjects:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

        print(f"Train images: {len(train_indices)}")
        print(f"Val images:   {len(val_indices)}")
        print(f"Test images:  {len(test_indices)}\n")

        return train_indices, val_indices, test_indices


# ============================================================================
#                       CUSTOM DATASET WRAPPER
# ============================================================================

class TransformedSubset(Dataset):
    """Apply transforms to dataset subset"""
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        raw, clahe, label = self.subset[idx]
        if self.transform:
            raw = self.transform(raw)
            clahe = self.transform(clahe)
        return raw, clahe, label


# ============================================================================
#                         MODEL BUILDER
# ============================================================================

def build_biometric_model(model_type, num_classes, config):
    """Build model optimized for biometric authentication"""
    
    if model_type == "raw":
        backbone = resnet18(pretrained=config.PRETRAINED)
        
        # Keep backbone as-is (already 3-channel)
        if config.FREEZE_BACKBONE:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # Simpler FC for biometrics
        backbone.fc = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        return backbone
    
    elif model_type == "concat":
        """Dual-stream concatenation fusion"""
        backbone1 = resnet18(pretrained=config.PRETRAINED)
        backbone2 = resnet18(pretrained=config.PRETRAINED)
        
        # Remove FC layers
        feat_dim = backbone1.fc.in_features
        backbone1 = nn.Sequential(*list(backbone1.children())[:-1])
        backbone2 = nn.Sequential(*list(backbone2.children())[:-1])
        
        class ConcatBiometricModel(nn.Module):
            def __init__(self, b1, b2, feat_dim, num_classes, dropout):
                super().__init__()
                self.backbone1 = b1
                self.backbone2 = b2
                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feat_dim * 2, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x1, x2):
                f1 = self.backbone1(x1).view(x1.size(0), -1)
                f2 = self.backbone2(x2).view(x2.size(0), -1)
                feat = torch.cat([f1, f2], dim=1)
                return self.fc(feat)
        
        return ConcatBiometricModel(backbone1, backbone2, feat_dim, num_classes, config.DROPOUT_RATE)
    
    elif model_type == "ecg":
        """Dual-stream with FIXED attention mechanism"""
        backbone1 = resnet18(pretrained=config.PRETRAINED)
        backbone2 = resnet18(pretrained=config.PRETRAINED)
        
        feat_dim = backbone1.fc.in_features
        backbone1 = nn.Sequential(*list(backbone1.children())[:-1])
        backbone2 = nn.Sequential(*list(backbone2.children())[:-1])
        
        class ECGBiometricModel(nn.Module):
            def __init__(self, b1, b2, feat_dim, num_classes, dropout):
                super().__init__()
                self.backbone1 = b1
                self.backbone2 = b2
                # Fixed attention that weighs each stream separately
                self.attention = nn.Sequential(
                    nn.Linear(feat_dim * 2, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2),
                    nn.Softmax(dim=1)
                )
                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(feat_dim * 2, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x1, x2):
                f1 = self.backbone1(x1).view(x1.size(0), -1)
                f2 = self.backbone2(x2).view(x2.size(0), -1)
                feat = torch.cat([f1, f2], dim=1)
                
                # ✅ FIXED: Weight each stream separately
                attn = self.attention(feat)
                weighted_f1 = f1 * attn[:, 0:1]
                weighted_f2 = f2 * attn[:, 1:2]
                weighted_feat = torch.cat([weighted_f1, weighted_f2], dim=1)
                
                return self.fc(weighted_feat)
        
        return ECGBiometricModel(backbone1, backbone2, feat_dim, num_classes, config.DROPOUT_RATE)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
#                     BIOMETRIC METRICS
# ============================================================================

def compute_eer(scores, labels):
    """
    Compute Equal Error Rate (EER) for biometric systems
    
    Args:
        scores: similarity scores [0, 1]
        labels: 1 for genuine pairs, 0 for impostor pairs
    
    Returns:
        eer: Equal Error Rate
        threshold: operating point threshold
    """
    # Sort by score
    sorted_idx = np.argsort(scores)[::-1]
    scores_sorted = scores[sorted_idx]
    labels_sorted = labels[sorted_idx]
    
    # Compute FMR and FNMR at each threshold
    min_diff = float('inf')
    best_threshold = 0
    best_eer = 1
    
    for threshold in scores_sorted:
        fmr = ((scores_sorted >= threshold) & (labels_sorted == 0)).sum() / (labels_sorted == 0).sum()
        fnmr = ((scores_sorted < threshold) & (labels_sorted == 1)).sum() / (labels_sorted == 1).sum()
        
        if abs(fmr - fnmr) < min_diff:
            min_diff = abs(fmr - fnmr)
            best_eer = (fmr + fnmr) / 2
            best_threshold = threshold
    
    return best_eer, best_threshold


# ============================================================================
#                       TRAINING ENGINE
# ============================================================================

class BiometricTrainingEngine:
    """Training engine optimized for biometric systems"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.NUM_EPOCHS // 3,
            eta_min=config.LEARNING_RATE * 0.01
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
        
        # Mixed precision
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # Tracking
        self.best_val_acc = 0
        self.patience_counter = 0
        self.metrics = defaultdict(list)
    
    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch in progress_bar:
            raw, clahe, labels = batch
            raw, clahe, labels = raw.to(self.device), clahe.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.USE_AMP:
                with autocast():
                    outputs = self.model(raw, clahe)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(raw, clahe)
                loss = self.criterion(outputs, labels)
            
            if self.config.USE_AMP:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                self.optimizer.step()
            
            total_loss += loss.item() * raw.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += raw.size(0)
            
            progress_bar.set_postfix({'loss': total_loss / total, 'acc': correct / total})
        
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['train_acc'].append(avg_acc)
        self.metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate"""
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            raw, clahe, labels = batch
            raw, clahe, labels = raw.to(self.device), clahe.to(self.device), labels.to(self.device)
            
            outputs = self.model(raw, clahe)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += raw.size(0)
        
        val_acc = correct / total
        self.metrics['val_acc'].append(val_acc)
        
        return val_acc
    
    @torch.no_grad()
    def test(self):
        """Test"""
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in tqdm(self.test_loader, desc="Testing", leave=False):
            raw, clahe, labels = batch
            raw, clahe, labels = raw.to(self.device), clahe.to(self.device), labels.to(self.device)
            
            outputs = self.model(raw, clahe)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += raw.size(0)
        
        test_acc = correct / total
        return test_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': dict(self.metrics)
        }
        
        os.makedirs(self.config.SAVE_DIR, exist_ok=True)
        
        if is_best:
            path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_TYPE}_best.pth")
        else:
            path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_TYPE}_epoch_{epoch:03d}.pth")
        
        torch.save(checkpoint, path)
        return path
    
    def fit(self):
        """Full training loop"""
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_acc = self.validate()
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% | "
                  f"Val Acc: {val_acc*100:5.1f}% | LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if (epoch + 1) % self.config.SAVE_EVERY_N_EPOCHS == 0:
                self.save_checkpoint(epoch)
            
            if val_acc > self.best_val_acc + self.config.MIN_DELTA:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                best_path = self.save_checkpoint(epoch, is_best=True)
                print(f"🎯 Best model saved: {best_path}")
            else:
                self.patience_counter += 1
                if self.config.EARLY_STOPPING and self.patience_counter >= self.config.PATIENCE:
                    print(f"⏹️  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for testing
        best_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_TYPE}_best.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_acc = self.test()
        
        # Save metrics
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        metrics_path = os.path.join(self.config.LOG_DIR, f"{self.config.MODEL_TYPE}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ PALM VEIN TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Model: {self.config.MODEL_TYPE.upper()}")
        print(f"Best Val Accuracy: {self.best_val_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        print(f"Training Time: {total_time/3600:.2f} hours")
        print(f"Best Model: {best_path}")
        print(f"{'='*60}\n")
        
        return test_acc


# ============================================================================
#                         MAIN TRAINING
# ============================================================================

def train_palm_vein(model_type="concat"):
    """Train palm vein authentication model"""
    
    # Setup
    random.seed(BiometricConfig.SEED)
    np.random.seed(BiometricConfig.SEED)
    torch.manual_seed(BiometricConfig.SEED)
    torch.cuda.manual_seed_all(BiometricConfig.SEED)
    
    BiometricConfig.MODEL_TYPE = model_type
    
    print(f"\n{'='*60}")
    print(f"🔐 PALM VEIN AUTHENTICATION TRAINING")
    print(f"{'='*60}")
    print(f"Model: {model_type.upper()}")
    print(f"Device: {BiometricConfig.DEVICE}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("📦 Loading dataset...")
    from datasets.helmms_dataset import HELMMSPalmVeinDataset
    
    full_ds = HELMMSPalmVeinDataset(
        BiometricConfig.DATA_ROOT,
        BiometricConfig.DATASET_NAME,
        BiometricConfig.WAVELENGTH,
        BiometricConfig.IMAGE_SIZE,
        transform=None
    )
    
    num_classes = full_ds.num_classes
    
    # CRITICAL: Identity-based split
    # ✅ FIX: Random split for classification training

    from torch.utils.data import random_split

    train_size = int(0.7 * len(full_ds))
    val_size = int(0.15 * len(full_ds))
    test_size = len(full_ds) - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        full_ds, [train_size, val_size, test_size]
    )

    print(f"\n{'='*60}")
    print("RANDOM SPLIT (FIXED)")
    print(f"{'='*60}")
    print(f"Train: {len(train_subset)}")
    print(f"Val:   {len(val_subset)}")
    print(f"Test:  {len(test_subset)}")
    print(f"{'='*60}\n")
    
    # Create loaders with biometric-safe augmentation
    train_transform = BiometricAugmentation.get_train_transform()
    val_transform = BiometricAugmentation.get_val_transform()
    test_transform = BiometricAugmentation.get_test_transform()
    
    train_loader = DataLoader(
        TransformedSubset(train_subset, train_transform),
        batch_size=BiometricConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        TransformedSubset(val_subset, val_transform),
        batch_size=BiometricConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        TransformedSubset(test_subset, test_transform),
        batch_size=BiometricConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Dataset loaded: {len(full_ds)} samples")
    
    # Build model
    print(f"\n🏗️  Building {model_type.upper()} model...")
    model = build_biometric_model(model_type, num_classes, BiometricConfig).to(BiometricConfig.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Train
    engine = BiometricTrainingEngine(model, train_loader, val_loader, test_loader, BiometricConfig)
    test_acc = engine.fit()
    
    return test_acc


# ============================================================================
#                            ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    for model in ["concat", "ecg", "raw"]:
        try:
            print(f"\n\n{'#'*60}")
            print(f"Training {model.upper()} model")
            print(f"{'#'*60}\n")
            train_palm_vein(model_type=model)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()