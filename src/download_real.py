from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AlexZigma/msr-vtt",
    repo_type="dataset",
    local_dir="./msr_vtt_data",
    allow_patterns=["data/MSR-VTT.ZIP", "data/test_videos.zip"],  # Skip checkpoint file
    ignore_patterns=["*checkpoint*.zip"]
)