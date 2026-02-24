#!/usr/bin/env python3
"""
extract_features.py — Optimized Parallel Version
Saves all outputs to /data:
 - features.npy, labels.npy
 - random_forest_model.pkl, xgboost_model.pkl, calibrated_classifier.pkl, scaler.pkl
"""

import os
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern
from scipy.fft import fft2, fftshift
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# ============================================================
# CONFIG
# ============================================================
EXPECTED_TOTAL_FEATURES = 2567
RESNET_FEATURES = 2048
HANDCRAFTED_FEATURES = EXPECTED_TOTAL_FEATURES - RESNET_FEATURES  # 519

DATA_DIR = Path("data")
PROCESSED_FRAMES_DIR = DATA_DIR / "processed_frames"
OUTPUT_FEATURES = DATA_DIR / "features.npy"
OUTPUT_LABELS = DATA_DIR / "labels.npy"
SCALER_PATH = DATA_DIR / "scaler.pkl"
RF_MODEL_PATH = DATA_DIR / "random_forest_model.pkl"
XGB_MODEL_PATH = DATA_DIR / "xgboost_model.pkl"
CALIBRATED_MODEL_PATH = DATA_DIR / "calibrated_classifier.pkl"

print(f"[CONFIG] Using total features: {EXPECTED_TOTAL_FEATURES}")

# ============================================================
# Load ResNet50 once globally
# ============================================================
try:
    base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    FEATURE_EXTRACTOR = Model(inputs=base_model.input, outputs=base_model.output)
    print("✅ ResNet50 loaded successfully")
except Exception as e:
    print(f"❌ Error loading ResNet50: {e}")
    FEATURE_EXTRACTOR = None


# ============================================================
# Feature extraction helper functions
# ============================================================
def ensure_uint8(image):
    if image is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def extract_multi_scale_lbp(image):
    """Multi-scale LBP — 260 features"""
    image = ensure_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    feats = []
    configs = [(8, 1, 10), (16, 2, 18), (24, 3, 26), (16, 4, 18)]
    for p, r, b in configs:
        lbp = local_binary_pattern(gray, p, r, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=b, range=(0, b), density=True)
        feats.extend(hist)
    feats = np.array(feats, dtype=np.float32)
    return np.pad(feats, (0, 260 - len(feats)))[:260]


def extract_edge_features(image):
    """Edge features — 100"""
    image = ensure_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    hist, _ = np.histogram(mag.ravel(), bins=100, range=(0, 255), density=True)
    return hist.astype(np.float32)


def extract_frequency_features(image):
    """Frequency features — 100"""
    gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_RGB2GRAY)
    fft_img = np.abs(fftshift(fft2(gray)))
    hist, _ = np.histogram(fft_img.ravel(), bins=100, range=(0, np.max(fft_img) + 1e-6), density=True)
    return hist.astype(np.float32)


def extract_color_features(image):
    """Color + intensity stats — 59"""
    image = ensure_uint8(image)
    feats = []
    for c in range(3):
        ch = image[:, :, c]
        feats.extend([np.mean(ch), np.std(ch), np.median(ch),
                      np.percentile(ch, 25), np.percentile(ch, 75),
                      np.min(ch), np.max(ch)])
        hist, _ = np.histogram(ch, bins=10, range=(0, 256), density=True)
        feats.extend(hist)
    feats.extend([0, 0, 0])
    return np.array(feats[:59], dtype=np.float32)


def extract_handcrafted_features(image):
    """Combine all handcrafted"""
    lbp = extract_multi_scale_lbp(image)
    edge = extract_edge_features(image)
    freq = extract_frequency_features(image)
    color = extract_color_features(image)
    features = np.concatenate([lbp, edge, freq, color])
    return np.pad(features, (0, HANDCRAFTED_FEATURES - len(features)))[:HANDCRAFTED_FEATURES]


# ============================================================
# Worker for parallel processing
# ============================================================
def process_frame_file(file_path, label):
    try:
        frame = np.load(file_path)
        if frame.shape != (224, 224, 3):
            return None
        deep = FEATURE_EXTRACTOR.predict(preprocess_input(np.expand_dims(frame, 0)), verbose=0)[0]
        hand = extract_handcrafted_features(frame)
        combined = np.concatenate([deep, hand])
        return combined, label
    except Exception:
        return None


# ============================================================
# Extract from folders
# ============================================================
def extract_features_from_folder(folder, label, max_workers=6):
    npy_files = sorted(Path(folder).rglob("*.npy"))
    print(f"📂 {folder}: {len(npy_files)} frames")

    all_features, all_labels = [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frame_file, f, label): f for f in npy_files}
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res:
                feat, lbl = res
                all_features.append(feat)
                all_labels.append(lbl)
            if (i + 1) % 500 == 0:
                print(f"   Processed {i+1}/{len(npy_files)}")

    X = np.array(all_features)
    y = np.array(all_labels)
    print(f"✅ Done: {X.shape[0]} frames, {X.shape[1]} features")
    return X, y


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    print("\n🚀 Starting Parallel Feature Extraction...")

    X_real, y_real = extract_features_from_folder(PROCESSED_FRAMES_DIR / "real", 0)
    X_fake, y_fake = extract_features_from_folder(PROCESSED_FRAMES_DIR / "fake", 1)

    X = np.vstack([X_real, X_fake])
    y = np.concatenate([y_real, y_fake])
    print(f"\nTotal frames combined: {X.shape[0]}, features: {X.shape[1]}")

    np.save(OUTPUT_FEATURES, X)
    np.save(OUTPUT_LABELS, y)
    print(f"✅ Saved → {OUTPUT_FEATURES}, {OUTPUT_LABELS}")

    # ========================================================
    # MODEL TRAINING
    # ========================================================
    print("\n⚙️ Training models...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Scaler saved")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(rf, RF_MODEL_PATH)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
    joblib.dump(xgb, XGB_MODEL_PATH)

    calibrated = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    calibrated.fit(X_train, y_train)
    joblib.dump(calibrated, CALIBRATED_MODEL_PATH)
    print("✅ Calibrated classifier saved")

    print("\n🎉 All features and models saved successfully to /data/")
