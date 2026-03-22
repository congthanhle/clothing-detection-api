from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import shutil, os

REPO_ID   = "Bingsu/adetailer"
FILENAME  = "deepfashion2_yolov8s-seg.pt"
LOCAL_DIR = "models"
LOCAL_PATH = os.path.join(LOCAL_DIR, FILENAME)

def download():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    if os.path.exists(LOCAL_PATH):
        print(f"[Model] Already exists at {LOCAL_PATH}, skipping download.")
    else:
        print("[Model] Downloading from HuggingFace...")
        cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        shutil.copy(cached, LOCAL_PATH)
        size_mb = os.path.getsize(LOCAL_PATH) / 1024 / 1024
        print(f"[Model] Saved to {LOCAL_PATH} ({size_mb:.1f} MB)")

def verify():
    print("[Model] Loading model for verification...")
    model = YOLO(LOCAL_PATH)
    print("[Model] Classes:", model.names)
    assert len(model.names) == 13, "Expected 13 DeepFashion2 classes"
    print("[Model] ✓ Verified — 13 classes confirmed")

if __name__ == "__main__":
    download()
    verify()
