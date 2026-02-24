# FactCheck — Deepfake Video Detection Using ResNet-50

![Python](https://img.shields.io/badge/Python-3.13.6-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey?logo=flask)
![React](https://img.shields.io/badge/React.js-Frontend-61DAFB?logo=react)
![License](https://img.shields.io/badge/License-Academic-green)

A hybrid machine learning system that combines deep learning features with handcrafted computer vision techniques to detect manipulated videos with high accuracy.

---

## 📌 Overview

FactCheck leverages **ResNet-50** for deep feature extraction combined with traditional computer vision features (LBP, edge detection, frequency analysis, color statistics) to identify subtle irregularities and artifacts in deepfake videos.

The system processes video frames through a comprehensive pipeline involving feature extraction, scaling, and ensemble classification to achieve robust detection performance.

---

## 🏗️ Technical Architecture

The system implements a hybrid approach combining:

| Feature Type | Description | Dimensions |
|---|---|---|
| **Deep Features** | ResNet-50 CNN feature vectors | 2048 |
| **LBP (Multi-scale)** | Local Binary Patterns at radii 1,2,3,4 | 260 |
| **Edge Features** | Sobel gradient magnitude & direction histograms | 100 |
| **Frequency Features** | FFT-based spectral analysis | 100 |
| **Color Features** | RGB channel statistics (mean, std, percentiles) | 59 |
| **Total** | Combined feature vector per frame | **2567** |

**Classification:** Dual ensemble approach using Random Forest and XGBoost with probability calibration.

---

## 📊 Performance Metrics

| Metric | Random Forest | XGBoost |
|---|---|---|
| **Accuracy** | 83.15% | 83.88% |
| **ROC-AUC** | 0.90 | 0.90 |
| **F1-Score (Fake)** | 0.84 | 0.84 |
| **Precision (Fake)** | 81% | 81% |
| **Recall (Fake)** | 87% | 88% |

- **Detection Threshold:** 1% (`FAKE_THRESHOLD = 0.01`)
- **Test Dataset Size:** 2,214 videos

---

## 🛠️ Tech Stack

### Backend
- **Python 3.13.6** — Core programming language
- **TensorFlow/Keras** — ResNet-50 feature extraction with pre-trained ImageNet weights
- **OpenCV (cv2)** — Video frame extraction and preprocessing
- **Scikit-Learn** — StandardScaler, model evaluation metrics
- **XGBoost** — Gradient boosting classifier
- **Flask** — REST API backend for video upload and prediction endpoints
- **PostgreSQL** — Relational database for storing analysis results
- **NumPy** — Numerical array operations
- **Joblib** — Model serialization and deserialization
- **Scikit-Image** — Handcrafted feature extraction (LBP, edge detection)

### Frontend
- **React.js** — Component-based UI framework
- **Chart.js & react-chartjs-2** — Timeline visualization and confidence score graphs
- **React Router** — Client-side routing
- **Node.js & npm** — Package management and build tools

---

## 💻 System Requirements

### Hardware
- **Processor:** AMD Ryzen 5 5500U or equivalent multi-core processor
- **RAM:** 16 GB DDR4 (minimum)
- **Storage:** 256 GB SSD (minimum)
- **Graphics:** Integrated Radeon Graphics or dedicated GPU (optional for acceleration)

### Software
- **OS:** Windows 11 / Windows 10 / Linux
- **Python:** 3.13.6 or higher
- **Node.js:** 14.x or higher
- **PostgreSQL:** Latest stable version
- **IDE:** VS Code or equivalent

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Apoorva744/FactCheck-deepfake_detection.git
cd FactCheck-deepfake_detection
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install tensorflow==2.13.0
pip install opencv-python==4.8.1
pip install scikit-learn==1.3.2
pip install xgboost==2.0.3
pip install flask==3.0.0
pip install flask-cors==4.0.0
pip install psycopg2==2.9.9
pip install numpy==1.24.3
pip install joblib==1.3.2
pip install scikit-image==0.22.0
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm install chart.js react-chartjs-2 react-router-dom
```

### 4. Database Configuration
```sql
CREATE DATABASE factcheck_db;

CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    filename VARCHAR(255),
    prediction VARCHAR(10),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 5. Model Files
Place the following trained model files in the `data/` directory:
- `calibrated_classifier.pkl` — Trained XGBoost/Random Forest with CalibratedClassifierCV wrapper
- `scaler.pkl` — Fitted StandardScaler for feature normalization

---

## ▶️ Usage

### Start Backend Server
```bash
python app.py
```
Runs on `http://localhost:5000`

### Start Frontend Server
```bash
cd frontend
npm start
```
Opens at `http://localhost:3000`

---

## 🔌 API Endpoints

### `POST /api/predict`
Upload a video for deepfake detection.

- **Content-Type:** `multipart/form-data`
- **Parameters:** Video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- **Response:** JSON with prediction, confidence score, and timeline data

---

## 🔄 Processing Pipeline

```
Video Input
    │
    ▼
1. Frame Extraction (OpenCV — max 50 frames, uniform sampling)
    │
    ▼
2. Preprocessing (Resize 224×224, RGB conversion, normalization)
    │
    ▼
3. Feature Extraction
    ├── Deep Features: ResNet-50 → 2048 dims
    └── Handcrafted: LBP + Edge + FFT + Color → 519 dims
    │
    ▼
4. Feature Scaling (StandardScaler — zero mean, unit variance)
    │
    ▼
5. Classification (Random Forest + XGBoost with calibration)
    │
    ▼
Output: REAL / FAKE + Confidence Score
```

---

## 🧠 Technical Justifications

**Why TensorFlow?**
Selected for production-ready deployment capabilities, optimized static computation graphs for inference-only workloads, and seamless Keras API integration with scikit-learn pipelines.

**Why XGBoost?**
Efficiently handles high-dimensional feature spaces (2567 dimensions) through L1/L2 regularization and tree pruning. Sequential gradient boosting corrects residual errors, achieving superior accuracy over parallel ensemble methods like Random Forest alone.

**Why Flask?**
Lightweight WSGI framework optimal for single-purpose video processing APIs without Django overhead. Synchronous request handling aligns with CPU-bound video processing tasks.

---

## 📂 Datasets

- **FaceForensics++ (C23 Compression)** — High-quality manipulated videos using Deepfakes, Face2Face, FaceSwap, NeuralTextures techniques
- **Deepfake Detection (DFD) Dataset** — Google's large-scale collection with controlled real and manipulated video pairs

---

## 🔮 Future Enhancements

- **Audio-Visual Synchronization** — Detect lip-sync mismatches and voice manipulation
- **Transformer-Based Models** — Implement Vision Transformers (ViT) and CLIP for improved feature representation
- **Real-Time Detection** — Edge deployment with model quantization for mobile/browser-based verification
- **Adversarial Robustness** — Continuous learning pipeline to adapt to emerging GAN architectures
- **Blockchain Integration** — Immutable media provenance tracking for authenticity verification

---

## 📄 License

Academic research project for educational purposes.

> **Note:** This system is designed for educational and research purposes. Production deployment should include additional security, scalability, and validation measures.
