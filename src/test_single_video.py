#!/usr/bin/env python3
"""
test_single_video.py - Test a specific video from training set
This will show if there's a feature extraction mismatch
"""

import numpy as np
import cv2
import joblib
from pathlib import Path
from predict import predict_video

def test_training_frame_vs_upload():
    """Compare features from training vs live prediction"""
    
    print("="*60)
    print("🧪 TESTING: Training Features vs Upload Features")
    print("="*60)
    
    # Load training features and labels
    try:
        X_train = np.load('data/features.npy')
        y_train = np.load('data/labels.npy')
        scaler = joblib.load('data/scaler.pkl')
        classifier = joblib.load('data/calibrated_classifier.pkl')
        print("✅ Loaded training data and models")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Get a FAKE sample from training
    fake_indices = np.where(y_train == 1)[0]
    if len(fake_indices) == 0:
        print("❌ No FAKE samples in training data!")
        return
    
    # Pick a random FAKE sample
    sample_idx = fake_indices[0]
    sample_features_train = X_train[sample_idx]
    
    print(f"\n📊 Training Sample #{sample_idx} (TRUE LABEL: FAKE)")
    print(f"   Feature shape: {sample_features_train.shape}")
    print(f"   First 5 features: {sample_features_train[:5]}")
    
    # Scale and predict
    sample_scaled = scaler.transform(sample_features_train.reshape(1, -1))
    prob = classifier.predict_proba(sample_scaled)[0, 1]
    pred = classifier.predict(sample_scaled)[0]
    
    print(f"\n🔮 Model Prediction on Training Sample:")
    print(f"   Predicted: {'FAKE' if pred == 1 else 'REAL'}")
    print(f"   FAKE probability: {prob:.4f}")
    
    if pred == 0:
        print(f"\n🚨 CRITICAL: Model predicts REAL for training FAKE sample!")
        print(f"   This means the model is broken or data is mislabeled!")
    else:
        print(f"\n✅ Model correctly predicts training sample as FAKE")
    
    # Now test with actual video prediction
    print("\n" + "="*60)
    print("🎬 TESTING: Upload a FAKE video from training folder")
    print("="*60)
    print("\nInstructions:")
    print("1. Find a video from your FAKE training folder")
    print("2. Note its filename")
    print("3. Upload it through the web interface")
    print("4. Compare the results")
    
    print("\n💡 Expected behavior:")
    print("   • Training sample: FAKE probability ~0.90+")
    print("   • Uploaded same video: Should also be ~0.90+")
    print("   • If very different: Feature extraction mismatch!")
    
def check_label_distribution():
    """Check if labels make sense"""
    print("\n" + "="*60)
    print("📋 LABEL DISTRIBUTION CHECK")
    print("="*60)
    
    try:
        y = np.load('data/labels.npy')
        
        real_count = np.sum(y == 0)
        fake_count = np.sum(y == 1)
        
        print(f"\nTraining labels:")
        print(f"   REAL (0): {real_count} samples")
        print(f"   FAKE (1): {fake_count} samples")
        
        if real_count == 0 or fake_count == 0:
            print("\n🚨 CRITICAL: One class is missing!")
            print("   Your labels might be wrong!")
        
        # Check processed frames folders
        real_frames = len(list(Path("data/processed_frames/real").glob("*.npy")))
        fake_frames = len(list(Path("data/processed_frames/fake").glob("*.npy")))
        
        print(f"\nProcessed frames folders:")
        print(f"   data/processed_frames/real: {real_frames} frames")
        print(f"   data/processed_frames/fake: {fake_frames} frames")
        
        if real_count != real_frames or fake_count != fake_frames:
            print("\n⚠️  WARNING: Mismatch between labels and frames!")
            print("   Labels might be assigned incorrectly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_specific_fake_video(video_path):
    """Test a specific FAKE video"""
    print("\n" + "="*60)
    print(f"🎬 TESTING SPECIFIC VIDEO: {video_path}")
    print("="*60)
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("\nRunning prediction...")
    result = predict_video(video_path)
    
    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
        return
    
    print("\n📊 PREDICTION RESULTS:")
    print(f"   Label: {result.get('label', 'N/A')}")
    print(f"   Suspicion (FAKE prob): {result.get('suspicion', 0):.4f}")
    print(f"   Confidence: {result.get('confidence', 0):.4f}")
    
    if result.get('label') == 'REAL' and result.get('suspicion', 0) < 0.5:
        print("\n🚨 PROBLEM CONFIRMED:")
        print("   Video from FAKE folder predicted as REAL!")
        print("\n💡 Possible causes:")
        print("   1. Feature extraction differs between training and prediction")
        print("   2. Video is not actually a deepfake (mislabeled)")
        print("   3. Model needs retraining with better data")
    else:
        print("\n✅ Video correctly identified as FAKE")

def main():
    test_training_frame_vs_upload()
    check_label_distribution()
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    print("\n1. Copy a video filename from data/videos/fake/")
    print("2. Run: python test_single_video.py <video_path>")
    print("3. Or upload through web interface and compare")
    print("\nExample:")
    print("  python test_single_video.py data/videos/fake/sample.mp4")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        test_specific_fake_video(video_path)
    else:
        main()