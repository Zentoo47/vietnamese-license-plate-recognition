# Flowchart Hoạt Động Chương Trình

Tài liệu này mô tả luồng hoạt động tổng thể của hệ thống **ITS License Plate Detection** từ lúc người dùng mở web, upload ảnh/video, hệ thống xử lý bằng AI, trả kết quả và lưu lịch sử vào database.

## 1. Flowchart Tổng Thể

```mermaid
flowchart TD
    A[Người dùng mở Web Dashboard] --> B[Chọn chức năng xử lý]
    B --> C{Loại dữ liệu upload?}

    C -- Ảnh --> D[Upload ảnh lên /api/detect]
    C -- Video --> E[Upload video lên /api/detect-video]

    D --> F[FastAPI kiểm tra định dạng ảnh]
    F --> G[OpenCV đọc ảnh thành ma trận pixel]
    G --> H[YOLOv8 phát hiện vùng biển số]
    H --> I[Crop vùng biển số theo bounding box]
    I --> J[Tiền xử lý ảnh biển số]
    J --> K[EasyOCR nhận dạng ký tự]
    K --> L[Format biển số Việt Nam]
    L --> M[Phân loại loại xe]
    M --> N[Vẽ bounding box và text lên ảnh]
    N --> O[Lưu kết quả vào SQLite]
    O --> P[Trả JSON và ảnh kết quả về frontend]
    P --> Q[Hiển thị kết quả trên Web Dashboard]

    E --> R[FastAPI kiểm tra định dạng video]
    R --> S[Lưu video gốc vào uploads/]
    S --> T[OpenCV mở video bằng VideoCapture]
    T --> U{Còn frame để đọc?}
    U -- Có --> V[Đọc frame tiếp theo]
    V --> W{Đến frame cần xử lý?}
    W -- Có --> X[YOLOv8 phát hiện biển số trong frame]
    X --> Y[Crop biển số và chạy EasyOCR]
    Y --> Z[Format text và phân loại loại xe]
    Z --> AA[Vẽ bounding box và text lên frame]
    W -- Không --> AB[Dùng kết quả cache gần nhất]
    AB --> AA
    AA --> AC[Ghi frame vào video kết quả]
    AC --> U
    U -- Không --> AD[Đóng VideoCapture và VideoWriter]
    AD --> AE[Lưu video kết quả vào results/]
    AE --> AF[Trả link video_result về frontend]
    AF --> AG[Frontend phát video kết quả]
```

## 2. Flowchart Xử Lý Ảnh

```mermaid
flowchart TD
    A[Upload ảnh] --> B[API /api/detect nhận file]
    B --> C{File có đúng định dạng?}
    C -- Không --> D[Trả lỗi File type not supported]
    C -- Có --> E[Đọc ảnh bằng OpenCV]
    E --> F{Ảnh đọc được không?}
    F -- Không --> G[Trả lỗi Cannot read image]
    F -- Có --> H[YOLOv8 phát hiện biển số]
    H --> I{Có phát hiện biển số?}
    I -- Không --> J[Trả kết quả không có biển số]
    I -- Có --> K[Crop từng vùng biển số]
    K --> L[EasyOCR đọc ký tự]
    L --> M[Chuẩn hóa format biển số]
    M --> N{Confidence quá thấp?}
    N -- Có --> O[Gán UNKNOWN]
    N -- Không --> P[Giữ biển số nhận diện]
    O --> Q[Vẽ kết quả lên ảnh]
    P --> Q
    Q --> R[Lưu record vào SQLite]
    R --> S[Trả JSON + ảnh base64]
    S --> T[Frontend hiển thị kết quả]
```

## 3. Flowchart Xử Lý Video

```mermaid
flowchart TD
    A[Upload video] --> B[API /api/detect-video nhận file]
    B --> C{File video hợp lệ?}
    C -- Không --> D[Trả lỗi file video không hỗ trợ]
    C -- Có --> E[Lưu video vào uploads/]
    E --> F[OpenCV mở video]
    F --> G{Mở video thành công?}
    G -- Không --> H[Trả lỗi Cannot open video file]
    G -- Có --> I[Khởi tạo VideoWriter]
    I --> J{Tạo video kết quả được không?}
    J -- Không --> K[Trả lỗi Cannot create result video]
    J -- Có --> L[Đọc từng frame]
    L --> M{Còn frame không?}
    M -- Không --> N[Đóng video và lưu kết quả]
    M -- Có --> O{Frame này cần detect?}
    O -- Có --> P[YOLOv8 phát hiện biển số]
    P --> Q[Crop biển số]
    Q --> R[EasyOCR nhận dạng ký tự]
    R --> S[Format biển số]
    S --> T[Vẽ bounding box và text]
    O -- Không --> U[Dùng detection cache gần nhất]
    U --> T
    T --> V[Ghi frame vào video output]
    V --> L
    N --> W[Lưu video vào results/]
    W --> X[Trả video_result về frontend]
    X --> Y[Web Dashboard hiển thị video]
```

## 4. Flowchart Lưu Lịch Sử

```mermaid
flowchart TD
    A[Có kết quả nhận diện] --> B[Lấy plate_text]
    B --> C[Lấy confidence]
    C --> D[Lấy vehicle_type]
    D --> E[Lấy processing_time]
    E --> F[Lấy bbox và thông tin ảnh]
    F --> G[Tạo DetectionRecord]
    G --> H[Lưu vào SQLite data/detections.db]
    H --> I[API /api/history đọc lịch sử]
    I --> J[Frontend hiển thị bảng lịch sử]
```

## 5. Giải Thích Ngắn Gọn

- **Frontend** cho phép người dùng upload ảnh hoặc video.
- **FastAPI** nhận request và điều phối xử lý.
- **OpenCV** đọc ảnh/video, crop biển số và ghi kết quả.
- **YOLOv8** phát hiện vị trí biển số trong ảnh hoặc frame video.
- **EasyOCR** đọc ký tự từ vùng biển số đã crop.
- **Regex/Formatter** chuẩn hóa biển số về định dạng Việt Nam.
- **SQLite** lưu lịch sử nhận diện để xem lại và thống kê.
- **Web Dashboard** hiển thị ảnh/video kết quả, danh sách biển số và lịch sử.
