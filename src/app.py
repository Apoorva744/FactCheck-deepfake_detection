# app.py - Deepfake Detection API (FIXED PREDICTION LOGIC)

from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import numpy as np
from pathlib import Path

# --- Authentication Imports ---
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask_bcrypt import Bcrypt

# --- Custom Modules ---
from predict import predict_video
from dbConnect import insert_result, register_user, get_user, fetch_history_by_user_id

# =================================================================
#                         FLASK SETUP
# =================================================================
app = Flask(__name__)

CORS(app, 
     supports_credentials=True,
     origins=["http://localhost:3000"],
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

app.config['SECRET_KEY'] = 'a_very_secret_key_for_your_deepfake_project'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False

bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# =================================================================
#                         USER MODEL
# =================================================================
class User(UserMixin):
    def __init__(self, id, username):
        self.id = str(id)
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    user_data = get_user(user_id=user_id)
    if user_data:
        return User(id=user_data[0], username=user_data[1])
    return None

# =================================================================
#                         AUTH ROUTES
# =================================================================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = register_user(username, password_hash)
    if user_id is False:
        return jsonify({"error": "Database connection error"}), 500
    if user_id is None:
        return jsonify({"error": "User already exists"}), 409

    user = User(id=user_id, username=username)
    login_user(user)
    return jsonify({"message": "Registration successful", "user_id": user_id, "username": username}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user_data = get_user(username=username)
    if user_data and bcrypt.check_password_hash(user_data[2], password):
        user = User(id=user_data[0], username=user_data[1])
        login_user(user)
        return jsonify({"message": "Login successful", "user_id": user.id, "username": user.username}), 200
    return jsonify({"error": "Invalid username or password"}), 401


@app.route('/api/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"}), 200


@app.route('/api/user_status', methods=['GET'])
def user_status():
    if current_user.is_authenticated:
        return jsonify({"isAuthenticated": True, "username": current_user.username, "user_id": current_user.id}), 200
    return jsonify({"isAuthenticated": False}), 200

# =================================================================
#                         HISTORY ROUTE
# =================================================================
@app.route('/api/history', methods=['GET'])
def get_history():
    if not current_user.is_authenticated:
        return jsonify([]), 200

    history = fetch_history_by_user_id(current_user.id)
    history_list = [{
        "id": item[0],
        "filename": item[1],
        "prediction": item[2],
        "confidence": float(item[3]),
        "uploaded_at": item[4].isoformat()
    } for item in history]
    return jsonify(history_list), 200

# =================================================================
#                         PREDICTION ROUTES
# =================================================================
@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint for Deepfake detection - FIXED VERSION."""
    print("\n" + "="*60)
    print("🎬 NEW PREDICTION REQUEST RECEIVED")
    print("="*60)
    
    try:
        print(f"📋 Request files keys: {list(request.files.keys())}")
        
        # Accept multiple file key names
        video_file = request.files.get('file') or request.files.get('video') or request.files.get('video_file')
        
        if not video_file:
            print("❌ No video file found in request")
            return jsonify({"error": "No file part in the request"}), 400
        
        if video_file.filename == '':
            print("❌ Empty filename")
            return jsonify({"error": "No selected file"}), 400

        video_filename = video_file.filename
        print(f"✅ Received file: {video_filename}")

        # Run prediction
        result = predict_video(video_file)

        if "error" in result:
            print(f"❌ PREDICTION FAILED: {result['error']}")
            return jsonify(result), 500

        # ================================================================
        # FIXED: Properly extract and interpret prediction results
        # ================================================================
        suspicion_score = result.get("suspicion", 0.0)  # Probability of FAKE (0-1)
        
        # Check if predict.py already provided a label
        if "label" in result:
            # Use the label from predict.py (preferred method)
            label = result["label"]
            confidence_score = result.get("confidence", 0.0)
            print(f"✅ Using label from predict.py: {label}")
        else:
            # Fallback: Calculate label from suspicion
            label = "FAKE" if suspicion_score > 0.5 else "REAL"
            # Calculate confidence based on prediction
            confidence_score = suspicion_score if label == "FAKE" else (1.0 - suspicion_score)
            print(f"✅ Calculated label from suspicion: {label}")
        
        # Convert to percentages for display
        suspicion_percent = round(suspicion_score * 100, 2)
        confidence_percent = round(confidence_score * 100, 2)
        
        timeline = result.get("timeline", [])

        print(f"📊 Results:")
        print(f"   Label: {label}")
        print(f"   Suspicion (FAKE prob): {suspicion_percent}%")
        print(f"   Confidence: {confidence_percent}%")

        # Save to database
        user_id = getattr(current_user, "id", None) if current_user.is_authenticated else None
        if user_id:
            print(f"💾 Saving result to DB for user {user_id}")
            insert_result(user_id, video_filename, label, confidence_score)
        else:
            print("ℹ️  No user logged in, skipping DB save")

        # Prepare response
        response = {
            "reportId": result.get("reportId"),
            "label": label,  # "FAKE" or "REAL"
            "suspicion": suspicion_score,  # 0-1 (probability of being FAKE)
            "confidence": confidence_score,  # 0-1 (confidence in prediction)
            "suspicion_percent": suspicion_percent,  # 0-100
            "confidence_percent": confidence_percent,  # 0-100
            "timeline": timeline,
            "peakTime": result.get("peakTime"),
            "facialStatus": result.get("facialStatus"),
            "facialDescription": result.get("facialDescription"),
            "audioVisualStatus": result.get("audioVisualStatus"),
            "audioVisualDescription": result.get("audioVisualDescription"),
            "technicalStatus": result.get("technicalStatus"),
            "technicalDescription": result.get("technicalDescription"),
        }
        
        print("="*60)
        print("✅ PREDICTION SUCCESS - Sending response")
        print("="*60 + "\n")
        
        return jsonify(response), 200

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ UNHANDLED EXCEPTION")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        print("=" * 60 + "\n")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict_alias():
    """Alias for /api/predict"""
    return predict()

# =================================================================
#                         HEALTH CHECK
# =================================================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    from predict import FEATURE_EXTRACTOR, CLASSIFIER, SCALER
    return jsonify({
        "status": "ok",
        "message": "Backend is running",
        "models_loaded": all([
            FEATURE_EXTRACTOR is not None,
            CLASSIFIER is not None,
            SCALER is not None
        ])
    }), 200

# =================================================================
#                         MAIN ENTRY
# =================================================================
if __name__ == "__main__":
    from predict import FEATURE_EXTRACTOR, CLASSIFIER, SCALER

    print("\n" + "="*60)
    print("🚀 STARTING DEEPFAKE DETECTION BACKEND")
    print("="*60)
    
    if CLASSIFIER and SCALER and FEATURE_EXTRACTOR:
        print("✅ All ML components confirmed ready.")
    else:
        print("⚠️  Some ML components missing:")
        print(f"   - Feature Extractor: {'✅' if FEATURE_EXTRACTOR else '❌'}")
        print(f"   - Classifier: {'✅' if CLASSIFIER else '❌'}")
        print(f"   - Scaler: {'✅' if SCALER else '❌'}")
        print("   Please run train.py first!")
    
    print("\n📡 Server starting on http://localhost:5000")
    print("   - Health check: http://localhost:5000/api/health")
    print("   - Prediction: http://localhost:5000/api/predict")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')