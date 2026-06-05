# Dataset cho Train Model Biển Số Xe Việt Nam

Tài liệu này hướng dẫn chuẩn bị dataset YOLO để train model phát hiện biển số xe Việt Nam.

## Cấu trúc thư mục

```text
src/dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img100.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── ...
│   └── val/
│       ├── img100.txt
│       └── ...
└── data.yaml
```

## Nguồn ảnh gợi ý

- Ảnh chụp thực tế từ đường phố/bãi xe/cổng ra vào.
- Dataset công khai trên Kaggle, HuggingFace hoặc Roboflow.
- Ảnh từ camera giao thông, camera an ninh hoặc camera điện thoại.

## Số lượng ảnh khuyến nghị

- Tối thiểu: 200-500 ảnh.
- Tốt: 500-1000 ảnh.
- Rất tốt: trên 1000 ảnh, đa dạng góc chụp và điều kiện ánh sáng.

## Label ảnh

Có thể dùng LabelImg hoặc Roboflow để gán nhãn bounding box.

```bash
pip install labelImg
labelImg src/dataset/images/train
```

## Format YOLO

Mỗi ảnh cần một file `.txt` cùng tên trong thư mục `labels`. Mỗi dòng biểu diễn một object:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Ví dụ:

```text
0 0.512345 0.623456 0.234567 0.156789
```

Trong đó:

- `0`: class id, với `0 = license_plate`.
- `x_center`, `y_center`: tọa độ tâm bbox, chuẩn hóa từ 0 đến 1.
- `width`, `height`: kích thước bbox, chuẩn hóa từ 0 đến 1.

## File data.yaml

Ví dụ cấu hình:

```yaml
path: src/dataset
train: images/train
val: images/val

names:
  0: license_plate
```

## Checklist trước khi train

- [ ] Đã thu thập đủ ảnh.
- [ ] Đã label toàn bộ ảnh.
- [ ] Đã chia tập train/val, thường dùng tỷ lệ 80/20.
- [ ] Đã kiểm tra label không bị lệch bbox hoặc sai class.
- [ ] Đã cập nhật `data.yaml` đúng đường dẫn.
- [ ] Đã loại ảnh trùng, ảnh quá mờ hoặc không có biển số rõ.

## Mẹo tăng chất lượng model

1. Đa dạng góc chụp: chính diện, nghiêng, xa/gần.
2. Đa dạng ánh sáng: ban ngày, ban đêm, mưa, ngược sáng.
3. Đa dạng phương tiện: xe máy, ô tô, taxi, xe tải, xe buýt.
4. Đa dạng loại biển: nền trắng, xanh, vàng; biển một dòng và hai dòng.
5. Kiểm tra thủ công một phần label sau khi augment hoặc import dataset.

## Train tham khảo

```bash
yolo detect train data=src/dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

Sau khi train xong, copy weight tốt nhất vào:

```text
src/trained_models/vietnamese_license_plate/weights/best.pt
```
