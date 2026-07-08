# Intelligent Traffic System (ITS) - License Plate Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Supported-yellow)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

Hệ thống nhận diện biển số xe thông minh (ALPR/ANPR) dành cho môi trường giao thông Việt Nam. Dự án cung cấp giải pháp toàn diện bao gồm: API nhận diện bằng **FastAPI**, giao diện Web Dashboard, tích hợp **YOLOv8**, **OpenCV** fallback, và trích xuất ký tự bằng **EasyOCR**.

---

## 🌟 Tính Năng Chính

- **Nhận Diện Ảnh (Image Detection):** Tải lên hình ảnh, tự động phát hiện và trích xuất biển số xe với độ chính xác cao.
- **Xử Lý Video (Video Processing):** Phân tích video theo từng frame, xuất video kết quả được vẽ bounding box và biển số.
- **OCR Chuyên Biệt (Vietnamese Plates):** Tiền xử lý hình ảnh và dùng regex định dạng lại chuẩn biển số Việt Nam (VD: `30A-123.45`).
- **Web Dashboard:** Giao diện người dùng trực quan, quản lý lịch sử nhận diện, thống kê lưu lượng và xem lại hình ảnh/video kết quả.
- **Fallback Cơ Chế Kép:** Tự động chuyển sang sử dụng OpenCV (Color Threshold & Contours) nếu không có mô hình YOLO chuyên dụng.
- **Lưu Trữ Lịch Sử:** Lưu vết mọi lượt quét với cơ sở dữ liệu SQLite cục bộ.

## 📁 Cấu Trúc Thư Mục

```text
ITS_License_Plate_Detection/
├── app/
│   ├── api/              # Định tuyến (Routes) API của FastAPI
│   ├── database/         # Tương tác DB (SQLAlchemy Models, Sessions)
│   ├── models/           # Logic AI: YOLO Detector và EasyOCR Recognizer
│   └── utils/            # Công cụ xử lý ảnh, format text, bounding box
├── data/                 # Thư mục lưu database SQLite (Tạo tự động khi chạy)
├── src/
│   ├── dataset/          # Dữ liệu huấn luyện, script thiết lập và data.yaml
│   └── trained_models/   # Script tải, huấn luyện và lưu trữ các file trọng số (.pt)
├── static/               # File tĩnh (CSS, JS) cho Frontend
├── templates/            # File HTML cho Web Dashboard (Jinja2)
├── main.py               # Entry-point để khởi chạy API và Web App
├── test_model.py         # Kịch bản kiểm tra model trên tập Validation / Video mẫu
└── requirements.txt      # Danh sách thư viện Python
```

## 🚀 Cài Đặt

### 1. Yêu cầu hệ thống
- Python 3.9 trở lên.
- Nên sử dụng GPU hỗ trợ CUDA (nếu có) để tăng tốc độ nhận diện của YOLO và EasyOCR.

### 2. Thiết lập môi trường

Clone repository và di chuyển vào thư mục dự án:
```bash
git clone https://github.com/your-username/ITS_License_Plate_Detection.git
cd ITS_License_Plate_Detection
```

Tạo và kích hoạt môi trường ảo (Virtual Environment):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

Cài đặt các gói phụ thuộc:
```bash
pip install -r requirements.txt
```

*(Lưu ý: Nếu sử dụng PyTorch với GPU, hãy đảm bảo tải đúng phiên bản PyTorch phù hợp với CUDA trên máy trước khi chạy các lệnh trên).*

## 💻 Cách Sử Dụng

### Khởi Chạy Web Server
Để khởi động hệ thống Backend và Frontend:
```bash
python main.py
```
Sau đó truy cập vào Web Dashboard tại: **[http://localhost:8000](http://localhost:8000)**

### Chạy qua Command Line (CLI) với một ảnh
Kiểm tra thử nghiệm nhanh trên console:
```bash
python main.py --image path\to\image.jpg
```

### Chạy Kịch Bản Kiểm Tra (Test Model)
Bạn có thể đánh giá mô hình trực tiếp qua script test cung cấp sẵn (chạy bằng YOLO/OpenCV):
```bash
python test_model.py
```

## 🔌 API Tham Khảo (Endpoints)

Dự án cung cấp REST API có thể tích hợp với các hệ thống khác:

| Phương thức | Endpoint | Mô tả |
| --- | --- | --- |
| `GET` | `/api/health` | Kiểm tra trạng thái hệ thống |
| `POST` | `/api/detect` | Upload ảnh và trả về JSON kết quả biển số xe |
| `POST` | `/api/detect-video` | Upload video, xử lý và trả về video kết quả |
| `GET` | `/api/history` | Lấy lịch sử nhận diện từ Database SQLite |
| `GET` | `/api/stats` | Thống kê số lượng lượt quét, thời gian phản hồi |

*Tài liệu API tương tác (Swagger UI) có tại:* **[http://localhost:8000/docs](http://localhost:8000/docs)**

## 🧠 Quản Lý Dữ Liệu & Huấn Luyện (Training)

Nếu bạn muốn huấn luyện lại mô hình YOLOv8 để bắt biển số chính xác hơn:

1. **Chuẩn bị dữ liệu:** Tham khảo `src/dataset/README_DATASET.md`.
2. **Setup Dataset:** Có thể chạy các script `setup_dataset.bat` (Windows) hoặc `.sh` (Linux) để chuẩn bị.
3. **Training:**
   ```bash
   python src/trained_models/train_model.py --action train --epochs 100
   ```
   *Model tốt nhất sẽ lưu tại `src/trained_models/vietnamese_license_plate/weights/best.pt`.*

## 📌 Ghi Chú Phát Triển

- Các thư mục runtime như `data/`, `results/`, `uploads/`, `debug_crops/`, và các file mô hình lớn `.pt` sẽ tự động bị bỏ qua khỏi Git (theo `.gitignore`).
- Nếu mô hình EasyOCR hoặc YOLO chạy lần đầu, quá trình tải weights từ internet có thể diễn ra.

---
*Phát triển bởi đội ngũ Hệ Thống Giao Thông Thông Minh (ITS).*
