# src/utils.py
import cv2
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model

# Load the pre-trained ResNet-50 model as a feature extractor
try:
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    FEATURE_EXTRACTOR = Model(inputs=base_model.input, outputs=base_model.output)
except Exception as e:
    print(f"Error loading ResNet-50 model: {e}")
    FEATURE_EXTRACTOR = None

def preprocess_frame(frame):
    """
    Resizes and normalizes an image frame.
    
    Args:
        frame (np.ndarray): The input image frame.
        
    Returns:
        np.ndarray: The preprocessed frame.
    """
    if frame is None:
        return None
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype("float32") / 255.0
    return frame

def batch_extract_frames_and_features(video_path, batch_size=32):
    """
    Extracts features from a video file in batches using the ResNet-50 model.
    
    Args:
        video_path (str): The path to the video file.
        batch_size (int): The number of frames to process in each batch.
        
    Returns:
        list: A list of feature vectors.
    """
    if FEATURE_EXTRACTOR is None:
        print("Feature extractor not loaded. Cannot process video.")
        return []

    features_list = []
    frames_batch = []
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        preprocessed_frame = preprocess_frame(frame)
        if preprocessed_frame is not None:
            frames_batch.append(preprocessed_frame)
            
            if len(frames_batch) == batch_size:
                features = FEATURE_EXTRACTOR.predict(np.array(frames_batch), verbose=0)
                features_list.extend(features)
                frames_batch = []
        
        frame_count += 1
    
    # Process any remaining frames in the last batch
    if frames_batch:
        features = FEATURE_EXTRACTOR.predict(np.array(frames_batch), verbose=0)
        features_list.extend(features)
        
    cap.release()
    print(f"Processed {frame_count} frames from {os.path.basename(video_path)}")
    return features_list