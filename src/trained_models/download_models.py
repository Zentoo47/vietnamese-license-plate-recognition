"""
Script download model license plate detection từ Ultralytics
"""

import urllib.request
import os

# URLs của các model có sẵn
MODELS = {
    # Model ANPR từ Ultralytics (demo model)
    "anpr_demo": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/anpr-demo-model.pt",
        "description": "Demo model cho video mẫu"
    },
    # YOLOv8n pretrained - dùng cho fine-tuning
    "yolov8n": {
        "url": "yolov8n.pt",
        "description": "YOLOv8 nano - baseline model"
    },
    # YOLOv8n-custom license plate
    "license_plate_yolov8n": {
        "url": "license-plate-detector-yolov8n.pt",
        "description": "YOLOv8n license plate detector"
    }
}

def download_file(url: str, output_path: str):
    """Download file với progress"""
    print(f"Downloading {url}...")
    print(f"Saving to {output_path}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_progress)
        print(f"\nDownloaded successfully!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def download_all_models():
    """Download tất cả models"""
    models_dir = os.path.dirname(os.path.abspath(__file__))

    for name, info in MODELS.items():
        output_file = os.path.join(models_dir, f"{name}.pt")

        # Skip if exists
        if os.path.exists(output_file):
            print(f"[SKIP] {name}.pt already exists")
            continue

        # Handle Ultralytics HUB models
        if info["url"].startswith("yolov8"):
            try:
                from ultralytics import YOLO
                print(f"Downloading {name} via ultralytics...")
                model = YOLO(info["url"])
                print(f"[OK] {name} downloaded")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
        else:
            download_file(info["url"], output_file)


if __name__ == "__main__":
    print("=" * 50)
    print("Downloading ANPR/License Plate Detection Models")
    print("=" * 50)
    download_all_models()
    print("\nDone!")
