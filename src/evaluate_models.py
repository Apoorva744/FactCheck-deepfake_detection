import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load saved data
# --------------------------
X = np.load("data/features.npy")
y = np.load("data/labels.npy")

scaler = joblib.load("data/scaler.pkl")
rf = joblib.load("data/random_forest_model.pkl")
xgb = joblib.load("data/xgboost_model.pkl")

# Scale features
X_scaled = scaler.transform(X)

# --------------------------
# Train-test split exactly like before
# --------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# RF Predictions
# --------------------------
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nClassification Report — Random Forest")
print(classification_report(y_test, rf_pred))

# Confusion Matrix RF
cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix — Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------
# XGBoost Predictions
# --------------------------
xgb_pred = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("\nClassification Report — XGBoost")
print(classification_report(y_test, xgb_pred))

# Confusion Matrix XGB
cm_xgb = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"])
plt.title("Confusion Matrix — XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------
# ROC Curves
# --------------------------

# RF
rf_probs = rf.predict_proba(X_test)[:, 1]
fpr1, tpr1, _ = roc_curve(y_test, rf_probs)
roc_auc1 = auc(fpr1, tpr1)

plt.figure()
plt.plot(fpr1, tpr1, label=f"RF AUC = {roc_auc1:.2f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve — Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# XGBoost
xgb_probs = xgb.predict_proba(X_test)[:, 1]
fpr2, tpr2, _ = roc_curve(y_test, xgb_probs)
roc_auc2 = auc(fpr2, tpr2)

plt.figure()
plt.plot(fpr2, tpr2, label=f"XGB AUC = {roc_auc2:.2f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve — XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
