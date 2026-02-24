from huggingface_hub import snapshot_download

# Download the entire dataset
snapshot_download(
    repo_id="faridlab/deepaction_v1",
    repo_type="dataset",
    local_dir="./ai_videos_dataset"
)

print("Download complete!")