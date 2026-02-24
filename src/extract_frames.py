# extract_frames.py
import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, frames_per_video=10):
    """
    Extracts a specified number of frames from a video and saves them as images.

    Args:
        video_path (str): The path to the input video file.
        output_dir (str): The directory where the extracted frames will be saved.
        frames_per_video (int): The number of frames to extract from the video.

    Returns:
        int: The number of frames successfully extracted.
    """
    # Create the output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Warning: Video {video_path} has 0 frames. Skipping.")
        cap.release()
        return 0
    
    # Calculate the step size to evenly sample frames
    # Use max(1, ...) to ensure at least one frame is always extracted if possible
    step = max(1, total_frames // frames_per_video)
    
    extracted_count = 0
    for i in range(frames_per_video):
        # Calculate the frame number to extract.
        # This approach ensures we get frames from the start, middle, and end of the video.
        frame_number = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        
        if not ret or frame is None:
            # Break if no more frames can be read
            break
            
        # Create a unique filename for the frame
        frame_filename = f"frame_{i:04d}.jpg"
        output_path = Path(output_dir) / frame_filename
        
        cv2.imwrite(str(output_path), frame)
        extracted_count += 1
        
    cap.release()
    return extracted_count


def batch_extract_frames(input_base_dir, output_base_dir, frames_per_video=10):
    """
    Batch processes videos in 'real' and 'fake' subdirectories to extract frames.

    Args:
        input_base_dir (str): The base directory containing 'real' and 'fake' video folders.
        output_base_dir (str): The base directory where extracted frames will be saved.
        frames_per_video (int): The number of frames to extract per video.
    """
    for folder in ["real", "fake"]:
        input_dir = Path(input_base_dir) / folder
        output_dir = Path(output_base_dir) / folder
        
        # Check if the input directory exists
        if not input_dir.exists():
            print(f"\nWarning: Input directory {input_dir} not found. Skipping.")
            continue
            
        print(f"\nProcessing {folder} videos from {input_dir}...")
        
        video_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ('.mp4', '.avi', '.mov')]
        
        if not video_files:
            print(f"No video files found in {input_dir}")
            continue
            
        for video_path in video_files:
            print(f"Processing {video_path.name}...")
            video_output_dir = output_dir / video_path.stem
            
            try:
                extracted = extract_frames(str(video_path), str(video_output_dir), frames_per_video)
                if extracted == 0:
                    print(f"Warning: 0 frames extracted from {video_path.name}")
            except Exception as e:
                print(f"Failed to process {video_path.name}: {e}")

if __name__ == "__main__":
    # Define the input and output directories relative to your project structure
    # This structure is crucial for the other scripts to work correctly
    INPUT_VIDEOS_DIR = "data/videos"
    OUTPUT_FRAMES_DIR = "data/raw_frames"
    
    print("--- Starting Frame Extraction Process ---")
    batch_extract_frames(
        input_base_dir=INPUT_VIDEOS_DIR,
        output_base_dir=OUTPUT_FRAMES_DIR,
        frames_per_video=15  # You can adjust this value
    )
    print("\n--- Frame Extraction Complete ---")