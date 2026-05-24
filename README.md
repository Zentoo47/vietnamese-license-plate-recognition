# Hệ Thống Nhận Diện Biển Số Xe Việt Nam (ITS)

Ứng dụng nhận diện biển số xe tự động sử dụng Deep Learning và OCR, được thiết kế cho hệ thống Giao thông Thông minh (ITS).

---

## Mục lục

- [Cài đặt](#1-cài-đặt)
- [Chạy ứng dụng](#2-chạy-ứng-dụng)
- [Train Model Riêng](#3-train-model-cho-biển-số-việt-nam)
- [Cấu trúc dự án](#4-cấu-trúc-dự-án)
- [API Endpoints](#5-api-endpoints)

---

## 1. Cài đặt

### Yêu cầu

- Python 3.8+
- RAM: 8GB+
- GPU (khuyến nghị): NVIDIA với CUDA để tăng tốc xử lý

### Các bước cài đặt

```bash
# 1. Clone hoặc tải project
cd ITS_License_Plate_Detection

# 2. Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

### Cài đặt Tesseract OCR (cần cho EasyOCR)

**Windows:**
1. Tải từ: https://github.com/UB-Mannheim/tesseract/wiki
2. Cài đặt và thêm vào PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-vie
```

---

## 2. Chạy Ứng dụng

### Chạy Web Server

```bash
python -X utf8 main.py --api
```

Sau đó mở trình duyệt: **http://localhost:8000**

### Chạy với ảnh cụ thể

```bash
python -X utf8 main.py --image duong_dan_anh.jpg
```

### Chạy với Video

```bash
python -X utf8 main.py --video duong_dan_video.mp4
```

---

## 3. Train Model Cho Biển Số Việt Nam

Để có độ chính xác cao, bạn nên train model riêng cho biển số VN.

### Bước 1: Tạo cấu trúc Dataset

```bash
python trained_models/train_model.py --action setup
```

Điều này sẽ tạo:
```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### Bước 2: Thu thập và Label Ảnh

1. **Thu thập ảnh:**
   - Chụp ảnh biển số xe thực tế
   - Tải dataset có sẵn từ Kaggle/HuggingFace
   - Đảm bảo đa dạng: góc chụp, ánh sáng, loại xe

2. **Label ảnh với LabelImg:**
   ```bash
   pip install labelImg
   labelImg dataset/images/train
   ```

3. **Format YOLO:**
   - Mỗi ảnh cần file `.txt` cùng tên
   - Format: `<class_id> <x_center> <y_center> <width> <height>`
   - Ví dụ: `0 0.512 0.623 0.234 0.156`

### Bước 3: Train Model

```bash
# Train với YOLOv8n (nhanh, nhẹ)
python trained_models/train_model.py --action train --model yolov8n --epochs 100

# Train với YOLOv8s (cân bằng)
python trained_models/train_model.py --action train --model yolov8s --epochs 100

# Train với YOLOv8m (chính xác hơn)
python trained_models/train_model.py --action train --model yolov8m --epochs 50
```

### Bước 4: Kiểm tra Model

```bash
python trained_models/train_model.py --action validate --weights trained_models/vietnamese_license_plate/weights/best.pt
```

### Mẹo để có Model Tốt

| Yếu tố | Khuyến nghị |
|---------|-------------|
| Số lượng ảnh | 500-2000 ảnh |
| Label chính xác | Kiểm tra kỹ bounding boxes |
| Đa dạng dữ liệu | Nhiều góc, ánh sáng, điều kiện |
| Validation set | 20% tổng ảnh |
| Epochs | 50-200 tùy dataset |

---

## 4. Cấu Trúc Dự Án

```
ITS_License_Plate_Detection/
├── models/
│   ├── __init__.py
│   ├── detector.py           # Phát hiện biển số (YOLO + OpenCV)
│   └── ocr_recognizer.py     # Nhận diện ký tự (EasyOCR)
├── utils/
│   ├── __init__.py
│   ├── image_processor.py    # Tiền xử lý ảnh
│   └── plate_utils.py        # Tiện ích xử lý biển số
├── api/
│   ├── __init__.py
│   └── routes.py             # FastAPI endpoints
├── database/
│   ├── __init__.py
│   └── models.py             # SQLAlchemy models
├── trained_models/
│   ├── download_models.py     # Script download model
│   └── train_model.py         # Script train YOLO
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html            # Giao diện web
├── data/                      # Database SQLite
├── uploads/                   # Ảnh upload
├── results/                   # Ảnh kết quả
├── main.py                    # File chạy chính
├── requirements.txt
└── README.md
```

---

## 5. API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| GET | `/` | Trang chủ |
| GET | `/api/health` | Health check |
| POST | `/api/detect` | Nhận diện biển số từ ảnh |
| GET | `/api/history` | Lịch sử nhận diện |
| GET | `/api/stats` | Thống kê |
| DELETE | `/api/history` | Xóa lịch sử |

### Ví dụ sử dụng API

```bash
# Upload ảnh để nhận diện
curl -X POST "http://localhost:8000/api/detect" \
  -F "file=@image.jpg"

# Lấy lịch sử
curl "http://localhost:8000/api/history?limit=10"

# Lấy thống kê
curl "http://localhost:8000/api/stats"
```

---

## Pipeline Xử Lý

```
Ảnh đầu vào
    │
    ▼
┌─────────────────┐
│ Preprocessing    │ ← Resize, Enhance Contrast
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Plate Detection  │ ← YOLO / OpenCV Color Detection
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Crop & Enhance  │ ← Crop vùng biển số
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ OCR Recognition  │ ← EasyOCR / Tesseract
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Post-processing  │ ← Format biển số VN
└─────────────────┘
    │
    ▼
Kết quả: "30A-123.45"
```

---

## Giấy phép

MIT License

---

## Đóng góp

Pull requests are welcome! Vui lòng tạo issue để báo lỗi hoặc đề xuất tính năng mới.
