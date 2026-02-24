#!/usr/bin/env python3
"""
deep_diagnosis.py - Deep dive into why model always predicts REAL
"""

import numpy as np
import joblib
from pathlib import Path

print("="*60)
print("🔬 DEEP DIAGNOSTIC - Model Behavior Analysis")
print("="*60)

# Load everything
try:
    X = np.load('data/features.npy')
    y = np.load('data/labels.npy')
    classifier = joblib.load('data/calibrated_classifier.pkl')
    scaler = joblib.load('data/scaler.pkl')
    print("✅ All files loaded")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    exit(1)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"REAL (0): {np.sum(y==0)}, FAKE (1): {np.sum(y==1)}")

# ================================================================
# TEST 1: Check what model predicts on training data
# ================================================================
print("\n" + "="*60)
print("TEST 1: Model Predictions on Training Data")
print("="*60)

X_scaled = scaler.transform(X)
predictions = classifier.predict(X_scaled)
probabilities = classifier.predict_proba(X_scaled)

print(f"\nPredictions breakdown:")
print(f"  Predicted REAL (0): {np.sum(predictions == 0)}")
print(f"  Predicted FAKE (1): {np.sum(predictions == 1)}")

print(f"\nProbability stats (FAKE class):")
print(f"  Mean: {np.mean(probabilities[:, 1]):.4f}")
print(f"  Median: {np.median(probabilities[:, 1]):.4f}")
print(f"  Min: {np.min(probabilities[:, 1]):.4f}")
print(f"  Max: {np.max(probabilities[:, 1]):.4f}")

if np.sum(predictions == 1) == 0:
    print("\n🚨 CRITICAL: Model NEVER predicts FAKE!")
    print("   This means the model is completely broken.")

# ================================================================
# TEST 2: Check calibrated vs base model
# ================================================================
print("\n" + "="*60)
print("TEST 2: Calibrated vs Base Model")
print("="*60)

# Get base classifier
if hasattr(classifier, 'base_estimator'):
    base_model = classifier.base_estimator
    print("✅ Found base model (before calibration)")
    
    base_predictions = base_model.predict(X_scaled)
    base_probabilities = base_model.predict_proba(X_scaled)
    
    print(f"\nBASE model predictions:")
    print(f"  Predicted REAL: {np.sum(base_predictions == 0)}")
    print(f"  Predicted FAKE: {np.sum(base_predictions == 1)}")
    print(f"  Mean FAKE prob: {np.mean(base_probabilities[:, 1]):.4f}")
    
    print(f"\nCALIBRATED model predictions:")
    print(f"  Predicted REAL: {np.sum(predictions == 0)}")
    print(f"  Predicted FAKE: {np.sum(predictions == 1)}")
    print(f"  Mean FAKE prob: {np.mean(probabilities[:, 1]):.4f}")
    
    if np.sum(base_predictions == 1) > 0 and np.sum(predictions == 1) == 0:
        print("\n⚠️  FOUND IT! Calibration broke the model!")
        print("   Base model works, but calibration makes it predict only REAL")
else:
    print("⚠️  Could not access base model")

# ================================================================
# TEST 3: Check actual REAL vs FAKE predictions
# ================================================================
print("\n" + "="*60)
print("TEST 3: Predictions by True Label")
print("="*60)

real_indices = np.where(y == 0)[0]
fake_indices = np.where(y == 1)[0]

real_predictions = predictions[real_indices]
fake_predictions = predictions[fake_indices]

real_probs = probabilities[real_indices, 1]
fake_probs = probabilities[fake_indices, 1]

print(f"\nFor TRUE REAL samples (label=0):")
print(f"  Predicted REAL: {np.sum(real_predictions == 0)}")
print(f"  Predicted FAKE: {np.sum(real_predictions == 1)}")
print(f"  Avg FAKE probability: {np.mean(real_probs):.4f}")

print(f"\nFor TRUE FAKE samples (label=1):")
print(f"  Predicted REAL: {np.sum(fake_predictions == 0)}")
print(f"  Predicted FAKE: {np.sum(fake_predictions == 1)}")
print(f"  Avg FAKE probability: {np.mean(fake_probs):.4f}")

if np.mean(real_probs) > 0.4 or np.mean(fake_probs) < 0.6:
    print("\n⚠️  Model is NOT learning to distinguish classes!")

# ================================================================
# TEST 4: Check feature scaling issues
# ================================================================
print("\n" + "="*60)
print("TEST 4: Feature Scaling Check")
print("="*60)

# Check if features are actually different after scaling
real_features_scaled = X_scaled[real_indices]
fake_features_scaled = X_scaled[fake_indices]

real_mean = np.mean(real_features_scaled, axis=0)
fake_mean = np.mean(fake_features_scaled, axis=0)

feature_diff = np.abs(real_mean - fake_mean)
print(f"\nScaled feature differences:")
print(f"  Mean difference: {np.mean(feature_diff):.6f}")
print(f"  Max difference: {np.max(feature_diff):.6f}")
print(f"  Features with diff > 0.1: {np.sum(feature_diff > 0.1)}")
print(f"  Features with diff > 0.5: {np.sum(feature_diff > 0.5)}")

if np.mean(feature_diff) < 0.01:
    print("\n⚠️  Features are TOO SIMILAR even after scaling!")
    print("   Your FAKE videos might not be real deepfakes")

# ================================================================
# TEST 5: Check if model is just predicting majority class
# ================================================================
print("\n" + "="*60)
print("TEST 5: Majority Class Baseline")
print("="*60)

majority_class = 0 if np.sum(y == 0) > np.sum(y == 1) else 1
majority_baseline = np.sum(y == majority_class) / len(y)

model_accuracy = np.sum(predictions == y) / len(y)

print(f"\nMajority class baseline: {majority_baseline:.2%}")
print(f"Model accuracy: {model_accuracy:.2%}")

if model_accuracy <= majority_baseline + 0.05:
    print("\n🚨 CRITICAL: Model is just predicting majority class!")
    print("   It hasn't learned anything meaningful.")

# ================================================================
# TEST 6: Sample 10 FAKE videos - what does model say?
# ================================================================
print("\n" + "="*60)
print("TEST 6: Sample Predictions on Known FAKES")
print("="*60)

if len(fake_indices) > 0:
    sample_size = min(10, len(fake_indices))
    sample_indices = np.random.choice(fake_indices, sample_size, replace=False)
    
    print(f"\nChecking {sample_size} known FAKE samples:")
    for i, idx in enumerate(sample_indices, 1):
        pred = predictions[idx]
        prob = probabilities[idx, 1]
        print(f"  Sample {i}: Predicted={'FAKE' if pred==1 else 'REAL'}, "
              f"FAKE prob={prob:.4f}")

# ================================================================
# DIAGNOSIS SUMMARY
# ================================================================
print("\n" + "="*60)
print("📋 DIAGNOSIS SUMMARY")
print("="*60)

issues = []

if np.sum(predictions == 1) == 0:
    issues.append("Model NEVER predicts FAKE class")

if np.mean(feature_diff) < 0.01:
    issues.append("Features too similar between REAL and FAKE")

if model_accuracy <= majority_baseline + 0.05:
    issues.append("Model just predicting majority class (no learning)")

if np.mean(fake_probs) < 0.6:
    issues.append("Model assigns low FAKE probability to TRUE FAKE samples")

if issues:
    print("\n🚨 PROBLEMS FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n💡 MOST LIKELY CAUSES:")
    print("   1. Your 'FAKE' videos are not actual deepfakes")
    print("   2. Model training parameters need adjustment")
    print("   3. Calibration is breaking the model")
    print("   4. Features not capturing deepfake artifacts")
    
    print("\n🔧 SOLUTIONS TO TRY:")
    print("   1. Verify your FAKE videos are AI-generated faces")
    print("   2. Retrain WITHOUT calibration (use base model)")
    print("   3. Try different model parameters")
    print("   4. Get proper deepfake dataset (FaceForensics++)")
else:
    print("\n✅ No critical issues found in model behavior")
    print("   Problem might be in prediction pipeline")

print("\n" + "="*60 + "\n")