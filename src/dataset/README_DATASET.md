# Dataset cho Train Model Biển Số Xe Việt Nam

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
