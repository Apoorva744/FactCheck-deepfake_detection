# train.py
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score

def main():
    features_path = Path("data/features.npy")
    labels_path = Path("data/labels.npy")
    if not features_path.exists() or not labels_path.exists():
        print("features.npy or labels.npy not found. Run extract_features.py first.")
        return
    X = np.load(features_path)
    y = np.load(labels_path)
    print("Loaded dataset:", X.shape, y.shape)
    if X.shape[0] == 0:
        print("Empty dataset")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    Path("data").mkdir(exist_ok=True)
    joblib.dump(scaler, "data/scaler.pkl")
    print("Saved scaler to data/scaler.pkl")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    rf.fit(X_train_s, y_train)
    y_rf = rf.predict(X_test_s)
    print("RF accuracy:", accuracy_score(y_test, y_rf))
    print(classification_report(y_test, y_rf))
    joblib.dump(rf, "data/random_forest_model.pkl")
    print("Saved data/random_forest_model.pkl")

    # XGBoost
    pos_weight = float(np.sum(y==0) / np.sum(y==1)) if np.sum(y==1)>0 else 1.0
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, scale_pos_weight=pos_weight)
    xgb.fit(X_train_s, y_train)
    y_xgb = xgb.predict(X_test_s)
    print("XGB accuracy:", accuracy_score(y_test, y_xgb))
    print(classification_report(y_test, y_xgb))
    joblib.dump(xgb, "data/xgboost_model.pkl")
    print("Saved data/xgboost_model.pkl")

    # Calibrate chosen model (use RF by default)
    chosen = xgb
    calibrated = CalibratedClassifierCV(chosen, method='sigmoid', cv='prefit')
    calibrated.fit(X_train_s, y_train)
    joblib.dump(calibrated, "data/calibrated_classifier.pkl")
    print("Saved calibrated classifier to data/calibrated_classifier.pkl")

if __name__ == "__main__":
    main()
