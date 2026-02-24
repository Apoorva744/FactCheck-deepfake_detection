import joblib, numpy as np

clf = joblib.load("data/calibrated_classifier.pkl")
scaler = joblib.load("data/scaler.pkl")

print("Classifier type:", type(clf))
print("n_features_in_:", getattr(clf, "n_features_in_", None))

X = np.load("data/features.npy")
y = np.load("data/labels.npy")

# Apply scaler
X_scaled = scaler.transform(X)
preds = clf.predict(X_scaled)
fake_prob = clf.predict_proba(X_scaled)[:, 1]

print("Training accuracy:", np.mean(preds == y))
print("Predicted fake ratio:", np.mean(preds))
print("Average fake probability:", np.mean(fake_prob))

import joblib, numpy as np
clf = joblib.load("data/calibrated_classifier.pkl")
scaler = joblib.load("data/scaler.pkl")
print("clf.n_features_in_:", getattr(clf, "n_features_in_", None))

