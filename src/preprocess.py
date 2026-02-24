# preprocess.py
import cv2
import os
import numpy as np
from pathlib import Path

def preprocess_frame(frame):
    """Resize to 224x224, convert to RGB, return uint8 array (0-255)."""
    if frame is None:
        return None
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # keep as uint8 (0-255) — do NOT divide by 255 here
    return frame.astype(np.uint8)

def process_folder(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    for root, _, files in os.walk(input_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = Path(root) / fname
                relative = Path(root).relative_to(input_path)
                out_dir = output_path / relative
                out_dir.mkdir(parents=True, exist_ok=True)
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"Warning: cannot read {img_path}, skipping.")
                    continue
                proc = preprocess_frame(frame)
                if proc is None:
                    continue
                np.save(out_dir / f"{img_path.stem}.npy", proc)
                processed_count += 1
    print(f"Preprocessing complete. Processed {processed_count} frames.")

if __name__ == "__main__":
    RAW_FRAMES_DIR = "data/raw_frames"
    PROCESSED_FRAMES_DIR = "data/processed_frames"
    print("--- Starting Frame Preprocessing ---")
    for subset in ["real", "fake"]:
        in_dir = Path(RAW_FRAMES_DIR) / subset
        out_dir = Path(PROCESSED_FRAMES_DIR) / subset
        if not in_dir.exists():
            print(f"Warning: {in_dir} not found — skipping.")
            continue
        print(f"Processing {in_dir} -> {out_dir}")
        process_folder(str(in_dir), str(out_dir))
    print("--- Preprocessing finished ---")
