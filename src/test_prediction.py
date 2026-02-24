# test_prediction.py - NO PCA VERSION

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
import joblib
import traceback
import tempfile

# optional handcrafted feature dependency
try:
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    print("⚠️ skimage not available — handcrafted features will be zeros.")

# -------------------------------
# Load ResNet50 (global)
# -------------------------------
try:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    FEATURE_EXTRACTOR = Model(inputs=base_model.input, outputs=base_model.output)
    print("✅ ResNet50 loaded")
except Exception as e:
    print("❌ Error loading ResNet50:", e)
    FEATURE_EXTRACTOR = None

# -------------------------------
# Handcrafted features (copied from predict.py)
# -------------------------------
def ensure_uint8(image):
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            return (image * 255).astype(np.uint8)
        return image.astype(np.uint8)
    return image.astype(np.uint8)

def extract_lbp_features(image, num_points=24, radius=8, bins=256):
    if not SKIMAGE_AVAILABLE:
        return np.zeros(bins, dtype=np.float32)
    image = ensure_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist.astype(np.float32)

def extract_fft_features(image, bins=256):
    image = ensure_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    maxv = np.max(mag) if np.max(mag) > 0 else 1.0
    hist, _ = np.histogram(mag.ravel(), bins=bins, range=(0, maxv), density=True)
    return hist.astype(np.float32)

def extract_color_stats(image):
    image = ensure_uint8(image)
    means = np.mean(image, axis=(0,1))
    stds = np.std(image, axis=(0,1))
    return np.concatenate([means, stds]).astype(np.float32)

def extract_edge_variance(image):
    image = ensure_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.array([np.var(edges)], dtype=np.float32)

def extract_handcrafted_features(image):
    lbp = extract_lbp_features(image)
    fft = extract_fft_features(image)
    color = extract_color_stats(image)
    edge = extract_edge_variance(image)
    return np.concatenate([lbp, fft, color, edge]).astype(np.float32)

# -------------------------------
# Feature extraction (copied from predict.py)
# -------------------------------
def extract_features_for_prediction(frames, batch_size=32):
    if FEATURE_EXTRACTOR is None:
        raise RuntimeError("Feature extractor not loaded")

    N = len(frames)
    deep_feats = []
    handcrafted_feats = []

    for start in range(0, N, batch_size):
        batch = frames[start:start+batch_size]
        arr = np.array(batch, dtype=np.float32)
        arr = preprocess_input(arr)
        preds = FEATURE_EXTRACTOR.predict(arr, verbose=0)
        deep_feats.append(preds)

    for img in frames:
        handcrafted_feats.append(extract_handcrafted_features(img))

    deep_feats = np.vstack(deep_feats) if deep_feats else np.zeros((0,2048), dtype=np.float32)
    handcrafted_feats = np.vstack(handcrafted_feats) if handcrafted_feats else np.zeros((0,0), dtype=np.float32)

    if handcrafted_feats.size == 0 and N > 0:
        handcrafted_feats = np.zeros((N, 519), dtype=np.float32)
    elif handcrafted_feats.size == 0 and N == 0:
        pass 

    combined = np.hstack([deep_feats, handcrafted_feats])
    return combined

# -------------------------------
# Prediction function (MODIFIED for No PCA)
# -------------------------------
def predict_video(file_or_path, frame_skip=10, batch_size=32):
    
    MODEL_PATH = Path("data/classifier.pkl")
    # PCA_PATH is intentionally ignored

    try:
        if not isinstance(file_or_path, (str, Path)):
             raise ValueError("Input must be a file path string for testing.")
        
        video_path = str(file_or_path)

        # load classifier
        if not MODEL_PATH.exists():
            return {"error": f"Classifier file missing: {MODEL_PATH}. Run train.py to create it."}
        classifier = joblib.load(MODEL_PATH)

        # PCA step is SKIPPED here. We proceed directly with full features.

        # extract frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Could not open video {video_path}"}

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                resized = cv2.resize(frame, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            frame_count += 1
        cap.release()

        if len(frames) == 0:
            return {"error": "No usable frames extracted from the video."}

        # extract features
        features_for_model = extract_features_for_prediction(frames, batch_size=batch_size) # Use full features

        # sanity check: classifier expects n_features_in_
        if hasattr(classifier, "n_features_in_"):
            needed = int(getattr(classifier, "n_features_in_"))
            have = features_for_model.shape[1]
            if needed != have:
                return {
                    "error": (
                        f"Feature dimension mismatch: classifier expects {needed} features "
                        f"but current input has {have}. Ensure train.py was run without PCA."
                    )
                }

        # run predictions
        probs = classifier.predict_proba(features_for_model)  # (N, C)
        
        # determine index for FAKE class (assuming 1=FAKE)
        classes = getattr(classifier, "classes_", [0, 1])
        fake_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else (probs.shape[1] - 1)

        # Calculate the frame-by-frame suspicion score (for the graph)
        timeline = [float(p[fake_idx]) for p in probs] 

        # FINAL SUSPICION: calculated from the mean of the timeline/fake probabilities
        suspicion = float(np.mean(timeline))  
        
        # Confidence logic is complex without real_idx, using the prediction max probability as confidence
        confidence = float(np.mean(np.max(probs, axis=1))) 
        peak_time = int(np.argmax(timeline) * frame_skip) if len(timeline) else 0

        return {
            "reportId": "TEST-REPORT-001",
            "suspicion": float(suspicion),
            "confidence": float(confidence),
            "timeline": timeline,
            "peakTime": peak_time,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

# -------------------------------
# Main Execution Block
# -------------------------------
if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python src/test_prediction.py <path_to_video_file>")
        print("\nExample: python src/test_prediction.py data/videos/fake_001.mp4")
        sys.exit(1)
        
    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        sys.exit(1)

    print(f"\n--- Running Analysis on: {video_path} (NO PCA) ---")
    
    result = predict_video(video_path)

    if "error" in result:
        print(f"❌ Analysis Failed: {result['error']}")
        sys.exit(1)

    # Extract results
    suspicion = result.get("suspicion", 0.0)
    confidence = result.get("confidence", 0.0)
    timeline = result.get("timeline", [])
    
    # Calculate key metrics
    suspicion_percent = round(suspicion * 100, 2)
    confidence_percent = round(confidence * 100, 2)
    peak_suspicion = round(max(timeline) * 100, 2) if timeline else 0.0
    
    print("\n--- RESULTS ---")
    print(f"✅ FINAL SUSPICION SCORE (Decimal): {suspicion}")
    print(f"✅ FINAL SUSPICION SCORE (Percent): {suspicion_percent}%")
    print(f"   Confidence Score: {confidence_percent}%")
    print(f"   Peak Suspicion in Timeline: {peak_suspicion}%")
    print(f"   Total Frames Analyzed: {len(timeline)}")
    
    if suspicion_percent == 0.0 and peak_suspicion > 0.0:
        print("\n🚨 CRITICAL WARNING: Mean suspicion calculated as 0% but peak is non-zero.")
        print("   The code is mathematically collapsing the average.")
    elif suspicion_percent > 0.0:
        print("\n🎉 SUCCESS: Non-zero suspicion confirmed in the backend!")
        print("   If the frontend still shows 0%, the issue is 100% in the frontend's JavaScript/display code.")