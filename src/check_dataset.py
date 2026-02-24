#!/usr/bin/env python3
"""
check_dataset.py - Diagnose why model predicts REAL for everything
"""

import numpy as np
import joblib
from pathlib import Path

print("="*60)
print("🔍 DATASET & MODEL DIAGNOSTIC")
print("="*60)

# ============================================================
# CHECK 1: Dataset Balance
# ============================================================
print("\n[CHECK 1] Dataset Balance")
print("-"*60)

try:
    X = np.load('data/features.npy')
    y = np.load('data/labels.npy')
    
    total = len(y)
    real_count = np.sum(y == 0)
    fake_count = np.sum(y == 1)
    
    print(f"Total samples: {total}")
    print(f"REAL samples (label=0): {real_count} ({real_count/total*100:.1f}%)")
    print(f"FAKE samples (label=1): {fake_count} ({fake_count/total*100:.1f}%)")
    
    if real_count == 0 or fake_count == 0:
        print("\n❌ CRITICAL: One class is missing!")
        print("   You need both REAL and FAKE videos in training data!")
    elif real_count / fake_count > 5 or fake_count / real_count > 5:
        ratio = max(real_count, fake_count) / min(real_count, fake_count)
        print(f"\n⚠️  SEVERE CLASS IMBALANCE: {ratio:.1f}:1 ratio")
        print("   This is likely causing your model to predict one class only!")
        print("\n   Solutions:")
        print("   1. Balance your dataset (equal REAL and FAKE)")
        print("   2. Or at least keep ratio under 2:1")
    else:
        print("\n✅ Dataset balance looks good")
        
    print(f"\nFeature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    
except FileNotFoundError:
    print("❌ Features or labels not found!")
    print("   Run: python extract_features.py")
    exit(1)

# ============================================================
# CHECK 2: Model Predictions on Training Data
# ============================================================
print("\n[CHECK 2] Model Behavior on Training Data")
print("-"*60)

try:
    classifier = joblib.load('data/calibrated_classifier.pkl')
    scaler = joblib.load('data/scaler.pkl')
    
    print("✅ Models loaded")
    
    # Test on sample of training data
    sample_size = min(100, len(X))
    sample_indices = np.random.choice(len(X), sample_size, replace=False)
    
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    # Scale and predict
    X_scaled = scaler.transform(X_sample)
    predictions = classifier.predict(X_scaled)
    probabilities = classifier.predict_proba(X_scaled)[:, 1]
    
    # Analyze predictions
    pred_real = np.sum(predictions == 0)
    pred_fake = np.sum(predictions == 1)
    
    print(f"\nPredictions on {sample_size} training samples:")
    print(f"  Predicted REAL: {pred_real} ({pred_real/sample_size*100:.1f}%)")
    print(f"  Predicted FAKE: {pred_fake} ({pred_fake/sample_size*100:.1f}%)")
    
    print(f"\nProbability distribution:")
    print(f"  Mean probability (FAKE): {np.mean(probabilities):.3f}")
    print(f"  Min probability: {np.min(probabilities):.3f}")
    print(f"  Max probability: {np.max(probabilities):.3f}")
    
    if pred_fake == 0:
        print("\n❌ PROBLEM FOUND: Model predicts REAL for ALL samples!")
        print("   This means the model didn't learn to distinguish classes.")
    elif pred_real == 0:
        print("\n⚠️  Model predicts FAKE for everything (labels might be swapped)")
    else:
        print("\n✅ Model makes varied predictions")
        
        # Check accuracy on sample
        correct = np.sum(predictions == y_sample)
        accuracy = correct / sample_size
        print(f"\nTraining sample accuracy: {accuracy:.2%}")
        
except FileNotFoundError as e:
    print(f"❌ Model files not found: {e}")
    print("   Run: python train.py")
    exit(1)

# ============================================================
# CHECK 3: Feature Statistics
# ============================================================
print("\n[CHECK 3] Feature Statistics")
print("-"*60)

# Compare REAL vs FAKE features
real_features = X[y == 0]
fake_features = X[y == 1]

if len(real_features) > 0 and len(fake_features) > 0:
    real_mean = np.mean(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)
    
    difference = np.abs(real_mean - fake_mean)
    avg_difference = np.mean(difference)
    
    print(f"Average feature difference (REAL vs FAKE): {avg_difference:.4f}")
    
    if avg_difference < 0.01:
        print("\n⚠️  WARNING: REAL and FAKE features are too similar!")
        print("   This means:")
        print("   1. Your FAKE videos might not be actual deepfakes")
        print("   2. Or feature extraction isn't capturing differences")
    else:
        print("\n✅ Features show meaningful differences between classes")
        
        # Find most discriminative features
        most_different = np.argsort(difference)[-5:]
        print(f"\nMost discriminative features (indices): {most_different}")

# ============================================================
# CHECK 4: Scaler Statistics
# ============================================================
print("\n[CHECK 4] Scaler Statistics")
print("-"*60)

try:
    print(f"Scaler mean (first 5 features): {scaler.mean_[:5]}")
    print(f"Scaler scale (first 5 features): {scaler.scale_[:5]}")
    
    # Check if scaler is doing anything
    if np.all(scaler.scale_ == 1.0):
        print("\n⚠️  WARNING: Scaler is not scaling features!")
    else:
        print("\n✅ Scaler is working correctly")
        
except Exception as e:
    print(f"Could not check scaler: {e}")

# ============================================================
# CHECK 5: Test Prediction on Known Samples
# ============================================================
print("\n[CHECK 5] Manual Prediction Test")
print("-"*60)

# Take one REAL and one FAKE sample
if len(real_features) > 0 and len(fake_features) > 0:
    test_real = real_features[0].reshape(1, -1)
    test_fake = fake_features[0].reshape(1, -1)
    
    # Scale and predict
    test_real_scaled = scaler.transform(test_real)
    test_fake_scaled = scaler.transform(test_fake)
    
    pred_real = classifier.predict(test_real_scaled)[0]
    prob_real = classifier.predict_proba(test_real_scaled)[0, 1]
    
    pred_fake = classifier.predict(test_fake_scaled)[0]
    prob_fake = classifier.predict_proba(test_fake_scaled)[0, 1]
    
    print(f"Known REAL sample → Predicted: {pred_real} (prob: {prob_real:.3f})")
    print(f"Known FAKE sample → Predicted: {pred_fake} (prob: {prob_fake:.3f})")
    
    if pred_real == 0 and pred_fake == 1:
        print("\n✅ Model correctly identifies known samples")
    elif pred_real == 1 and pred_fake == 0:
        print("\n⚠️  Labels might be SWAPPED!")
        print("   Check if label 0=FAKE, 1=REAL instead of 0=REAL, 1=FAKE")
    else:
        print("\n⚠️  Model predictions are inconsistent")

# ============================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================
print("\n" + "="*60)
print("📋 DIAGNOSIS SUMMARY")
print("="*60)

issues = []
solutions = []

# Check class balance
if 'real_count' in locals() and 'fake_count' in locals():
    if real_count == 0 or fake_count == 0:
        issues.append("Missing REAL or FAKE class")
        solutions.append("Add videos for the missing class")
    elif max(real_count, fake_count) / min(real_count, fake_count) > 3:
        issues.append(f"Severe class imbalance ({max(real_count, fake_count)/min(real_count, fake_count):.1f}:1)")
        solutions.append("Balance dataset to 1:1 or at least 2:1 ratio")

# Check model predictions
if 'pred_fake' in locals() and pred_fake == 0:
    issues.append("Model always predicts REAL")
    solutions.append("Retrain with balanced data and check labels")

# Check feature differences
if 'avg_difference' in locals() and avg_difference < 0.01:
    issues.append("REAL and FAKE features too similar")
    solutions.append("Verify FAKE videos are actual deepfakes, not just edited videos")

if issues:
    print("\n⚠️  Issues Found:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n💡 Recommended Solutions:")
    for i, solution in enumerate(solutions, 1):
        print(f"   {i}. {solution}")
else:
    print("\n✅ No obvious issues found in dataset/model")
    print("\n   If predictions are still wrong, the problem might be:")
    print("   • Feature extraction differs between training and prediction")
    print("   • Test videos are different type than training videos")

print("\n" + "="*60 + "\n")