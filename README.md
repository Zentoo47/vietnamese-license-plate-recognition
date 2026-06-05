# ITS License Plate Detection

Ứng dụng nhận diện biển số xe dùng FastAPI, YOLOv8/OpenCV và EasyOCR. Dự án hỗ trợ nhận diện từ ảnh, xử lý video, lưu lịch sử vào SQLite và hiển thị kết quả trên giao diện web.

## Tính năng chính

- Upload ảnh và nhận diện biển số xe.
- Upload video và xuất video đã vẽ bounding box/kết quả nhận diện.
- OCR biển số bằng EasyOCR, có bước tiền xử lý và chuẩn hóa định dạng biển số Việt Nam.
- Lưu lịch sử nhận diện, confidence, thời gian xử lý và đường dẫn kết quả vào SQLite.
- Giao diện web có tab nhận diện ảnh, video, lịch sử và thống kê.
- Có fallback OpenCV khi chưa có model YOLO chuyên dụng.

## Cấu trúc dự án

```text
.
├── app/
│   ├── api/              # FastAPI routes
│   ├── database/         # SQLAlchemy models/session
│   ├── models/           # Detector và OCR recognizer
│   └── utils/            # Tiện ích xử lý ảnh/biển số
├── data/                 # SQLite DB runtime (không commit)
├── results/              # Video kết quả runtime (không commit)
├── static/               # CSS/JS frontend
├── templates/            # HTML frontend
├── uploads/              # File upload runtime (không commit)
├── src/dataset/          # Dataset config và hướng dẫn train
├── main.py               # Entry point web/API/CLI
└── requirements.txt
```

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Nếu dùng GPU, hãy cài đúng bản PyTorch/CUDA phù hợp với máy trước khi chạy EasyOCR/YOLO.

## Chạy ứng dụng

```bash
python main.py
```

Sau đó mở trình duyệt tại:

```text
http://localhost:8000
```

## Chạy CLI với một ảnh

```bash
python main.py --image path\to\image.jpg
```

## API chính

- `GET /api/health`: kiểm tra trạng thái API.
- `POST /api/detect`: upload ảnh và nhận diện biển số.
- `POST /api/detect-video`: upload video và xử lý từng frame.
- `GET /api/results/{filename}`: tải ảnh/video kết quả.
- `GET /api/history`: xem lịch sử nhận diện.
- `GET /api/stats`: xem thống kê tổng quan.
- `DELETE /api/history`: xóa lịch sử.

## Model và dữ liệu

Các file model `.pt`, database, ảnh/video upload và kết quả runtime không nên commit vào Git. Hãy đặt model tùy chỉnh tại một trong các vị trí ưu tiên sau:

```text
src/trained_models/license_plate_yolov8n.pt
src/trained_models/vietnamese_license_plate/weights/best.pt
src/trained_models/anpr_demo.pt
```

Nếu chưa có model chuyên dụng, detector sẽ dùng fallback OpenCV.

## Ghi chú phát triển

- Dataset và hướng dẫn train nằm trong `src/dataset/README_DATASET.md`.
- Cập nhật `.gitignore` trước khi thêm dữ liệu lớn.
- Không commit `data/`, `uploads/`, `results/`, `debug_crops/`, cache Python hoặc file model lớn.
