"""
Script train YOLO cho nhận diện biển số xe Việt Nam
Hướng dẫn chi tiết cách train model riêng
"""

import os
import shutil
import yaml
from datetime import datetime

# Cấu hình
PROJECT_NAME = "vietnamese_license_plate"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
MODEL_DIR = os.path.join(PROJECT_DIR, "trained_models")


def create_dataset_structure():
    """Tạo cấu trúc thư mục dataset cho YOLO"""
    print("Creating dataset structure...")

    folders = [
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "images", "val"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "labels", "val"),
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Tạo file YAML cấu hình
    data_yaml = {
        'path': DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'license_plate'
        },
        'nc': 1
    }

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Dataset structure created at: {DATASET_DIR}")
    print(f"Config file: {yaml_path}")
    return yaml_path


def create_sample_images_readme():
    """Tạo README hướng dẫn thu thập ảnh"""
    readme_content = """# Dataset cho Train Model Biển Số Xe Việt Nam

## Cấu trúc thư mục

```
dataset/
├── images/
│   ├── train/          # Ảnh train (80%)
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/            # Ảnh validation (20%)
│       ├── img100.jpg
│       └── ...
├── labels/
│   ├── train/          # Label YOLO cho train
│   │   ├── img001.txt
│   │   └── ...
│   └── val/            # Label YOLO cho validation
│       ├── img100.txt
│       └── ...
└── data.yaml
```

## Cách thu thập ảnh

### 1. Nguồn ảnh
- Chụp ảnh thực tế từ đường phố
- Tải từ dataset có sẵn trên Kaggle/HuggingFace
- Sử dụng ảnh từ camera giao thông

### 2. Dataset có sẵn tham khảo
- Kaggle: "Vietnamese License Plate Dataset"
- HuggingFace: Các dataset ANPR cho Đông Nam Á
- Roboflow: Vietnamese Vehicle License Plates

### 3. Số lượng ảnh khuyến nghị
- Tối thiểu: 200-500 ảnh
- Tốt: 500-1000 ảnh
- Xuất sắc: 1000+ ảnh

## Cách Label ảnh

### Sử dụng LabelImg
```bash
pip install labelImg
labelImg dataset/images/train
```

### Format YOLO
Mỗi file `.txt` chứa thông tin bounding box cho mỗi object trong ảnh:
```
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:
```
0 0.512345 0.623456 0.234567 0.156789
```

Trong đó:
- `0`: class_id (0 = license_plate)
- `0.512345`: x_center (tọa độ tâm X, từ 0 đến 1)
- `0.623456`: y_center (tọa độ tâm Y, từ 0 đến 1)
- `0.234567`: width (chiều rộng, từ 0 đến 1)
- `0.156789`: height (chiều cao, từ 0 đến 1)

## Checklist trước khi train

- [ ] Đã thu thập đủ ảnh (>200 ảnh)
- [ ] Đã label tất cả ảnh
- [ ] Đã chia train/val (80/20)
- [ ] Đã kiểm tra file labels không bị lỗi
- [ ] Đã tạo file data.yaml

## Mẹo

1. **Đa dạng góc chụp**: Chụp từ nhiều góc khác nhau
2. **Điều kiện ánh sáng**: Ban ngày, ban đêm, trời mưa
3. **Loại xe**: Xe máy, ô tô con, xe tải, xe buýt
4. **Khoảng cách**: Gần, trung bình, xa
"""

    readme_path = os.path.join(DATASET_DIR, "README_DATASET.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"Sample README created: {readme_path}")


def train_model(model_name: str = "yolov8n", epochs: int = 100, batch_size: int = 16, device: str = None):
    """
    Train YOLO model cho license plate detection

    Args:
        model_name: Tên model base (yolov8n, yolov8s, yolov8m)
        epochs: Số epochs để train
        batch_size: Batch size
        device: Thiết bị ('cpu', '0', '0,1,2,3' cho GPU)
    """
    from ultralytics import YOLO

    yaml_path = os.path.join(DATASET_DIR, "data.yaml")

    if not os.path.exists(yaml_path):
        print("ERROR: data.yaml not found. Run create_dataset_structure() first!")
        return

    print(f"Starting training with {model_name}...")
    print(f"Dataset: {yaml_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

    # Load pretrained model
    model = YOLO(f'{model_name}.pt')

    # Auto-detect device if not specified
    if device is None:
        import torch
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")

    # Train
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        project=MODEL_DIR,
        name=PROJECT_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=50,
        save=True,
        save_period=10,
        cache=True,
        device=device,
        workers=8,
        verbose=True
    )

    # Export model
    best_model_path = os.path.join(MODEL_DIR, PROJECT_NAME, "weights", "best.pt")
    if os.path.exists(best_model_path):
        print(f"\nTraining complete! Best model: {best_model_path}")

        # Export to ONNX
        model.export(format='onnx')
        print(f"Exported to ONNX format")

    return results


def validate_model(model_path: str):
    """Validate trained model"""
    from ultralytics import YOLO

    print(f"Validating model: {model_path}")
    model = YOLO(model_path)

    # Validate
    results = model.val(data=os.path.join(DATASET_DIR, "data.yaml"))

    print(f"\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")

    return results


def export_model(model_path: str, format: str = 'onnx'):
    """Export model sang format khác"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    exported_path = model.export(format=format)
    print(f"Model exported to: {exported_path}")
    return exported_path


def quick_test_with_pretrained():
    """Test nhanh với model đã train sẵn (fine-tune từ YOLOv8)"""
    from ultralytics import YOLO
    import cv2

    print("Testing with YOLOv8n (will auto-download if needed)...")

    # Load model
    model = YOLO('yolov8n.pt')

    # Test image (placeholder - bạn cần thay bằng ảnh thật)
    test_img = cv2.imread("test_image.jpg")

    if test_img is not None:
        results = model(test_img)
        print(f"Detected {len(results[0].boxes)} objects")

    print("\nĐể có model tốt hơn cho biển số VN, bạn cần:")
    print("1. Thu thập dataset biển số VN")
    print("2. Label ảnh với LabelImg")
    print("3. Train model với: python train_model.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO License Plate Detection Model")
    parser.add_argument("--action", type=str, default="setup",
                       choices=["setup", "train", "validate", "export", "test"],
                       help="Action to perform")
    parser.add_argument("--model", type=str, default="yolov8n",
                       help="Base model (yolov8n, yolov8s, yolov8m)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--weights", type=str, default=None,
                       help="Path to trained weights for validation/export")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: 'cpu' for CPU, '0' for GPU 0, '0,1,2,3' for multiple GPUs")

    args = parser.parse_args()

    if args.action == "setup":
        print("=" * 50)
        print("SETUP: Creating dataset structure")
        print("=" * 50)
        create_dataset_structure()
        create_sample_images_readme()
        print("\nSetup complete!")
        print("\nTiếp theo:")
        print("1. Thêm ảnh vào dataset/images/train/")
        print("2. Label ảnh với: labelImg dataset/images/train")
        print("3. Train: python train_model.py --action train")

    elif args.action == "train":
        print("=" * 50)
        print("TRAINING: Training YOLO model")
        print("=" * 50)
        train_model(args.model, args.epochs, args.batch, args.device)

    elif args.action == "validate":
        if args.weights:
            validate_model(args.weights)
        else:
            print("ERROR: --weights required for validation")

    elif args.action == "export":
        if args.weights:
            export_model(args.weights)
        else:
            print("ERROR: --weights required for export")

    elif args.action == "test":
        quick_test_with_pretrained()
