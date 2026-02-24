from pathlib import Path

real = Path("data/processed_frames/real")
fake = Path("data/processed_frames/fake")

print("Real npy files:", len(list(real.rglob("*.npy"))))
print("Fake npy files:", len(list(fake.rglob("*.npy"))))
