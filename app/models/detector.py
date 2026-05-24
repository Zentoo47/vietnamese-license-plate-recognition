"""
Module phát hiện biển số xe - Sử dụng model chuyên dụng + fallback
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import logging

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Phát hiện biển số xe - Sử dụng:
    1. Model chuyên dụng (license plate detector)
    2. YOLOv8 fine-tuned (nếu có)
    3. OpenCV color-based detection (fallback)
    """

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.5,
                 max_image_size: int = 800):
        """
        Khởi tạo detector

        Args:
            model_path: Đường dẫn đến model YOLO (.pt file)
            conf_threshold: Ngưỡng confidence để chấp nhận detection
            max_image_size: Kích thước tối đa ảnh đầu vào
        """
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_path = model_path or self._get_best_model_path()
        self.max_image_size = max_image_size
        self._model_loaded = False
        self._model_type = None  # 'license_plate', 'yolo', 'opencv'

    def _get_best_model_path(self) -> Optional[str]:
        """Tìm model tốt nhất có sẵn"""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "trained_models")

        # Ưu tiên model chuyên dụng
        priority_models = [
            "license_plate_yolov8n.pt",
            "vietnamese_license_plate/weights/best.pt",
            "anpr_demo.pt",
        ]

        for model in priority_models:
            path = os.path.join(models_dir, model)
            if os.path.exists(path):
                logger.info(f"Found pretrained model: {path}")
                return path

        return None

    def _load_model(self):
        """Load YOLO model"""
        if self._model_loaded:
            return

        try:
            from ultralytics import YOLO
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"YOLO device: {device}")

            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading custom model: {self.model_path}")
                self.model = YOLO(self.model_path)
                self._model_type = 'license_plate'
            else:
                # Thử download demo model
                logger.info("No custom model found. Using OpenCV fallback.")
                self._model_type = 'opencv'

            if self.model:
                self.model.to(device)

            self._model_loaded = True

        except ImportError:
            logger.warning("ultralytics not installed. Using OpenCV detection.")
            self._model_type = 'opencv'
            self._model_loaded = True

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh"""
        h, w = image.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image

    def _detect_by_color(self, image: np.ndarray) -> List[dict]:
        """
        Phát hiện biển số dựa trên màu sắc - cho biển số VN
        """
        detections = []
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ==== Biển số nền XANH chữ TRẮNG ====
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_blue:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            area = cw * ch

            if 1.5 < aspect < 5.0 and 500 < area < w * h * 0.1:
                detections.append({
                    'bbox': (x, y, x + cw, y + ch),
                    'confidence': 0.85,
                    'class_id': 0,
                    'class_name': 'blue_plate'
                })

        # ==== Biển số nền TRẮNG chữ ĐEN ====
        _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours_white, _ = cv2.findContours(thresh_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_white:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect = cw / ch if ch > 0 else 0
            area = cw * ch

            if 1.5 < aspect < 5.0 and 500 < area < w * h * 0.1:
                roi = gray[y:y+ch, x:x+cw]
                if roi.size > 0 and np.mean(roi) > 150:
                    detections.append({
                        'bbox': (x, y, x + cw, y + ch),
                        'confidence': 0.80,
                        'class_id': 1,
                        'class_name': 'white_plate'
                    })

        # ==== Edge detection ====
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(blurred, 30, 200)
        contours_edge, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_edge = sorted(contours_edge, key=cv2.contourArea, reverse=True)[:15]

        for contour in contours_edge:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect = cw / ch if ch > 0 else 0
                area = cw * ch

                if 1.5 < aspect < 5.0 and 500 < area < w * h * 0.1:
                    detections.append({
                        'bbox': (x, y, x + cw, y + ch),
                        'confidence': 0.70,
                        'class_id': 2,
                        'class_name': 'edge_plate'
                    })

        return self._simple_nms(detections)

    def _simple_nms(self, detections: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        """Simple NMS"""
        if len(detections) <= 1:
            return detections

        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        keep = []

        for det in detections:
            is_dup = False
            x1, y1, x2, y2 = det['bbox']
            area1 = (x2-x1) * (y2-y1)

            for kept in keep:
                kx1, ky1, kx2, ky2 = kept['bbox']
                xi, yi = max(x1, kx1), max(y1, ky1)
                x2i, y2i = min(x2, kx2), min(y2, ky2)

                if xi < x2i and yi < y2i:
                    inter = (x2i - xi) * (y2i - yi)
                    area2 = (kx2-kx1) * (ky2-ky1)
                    union = area1 + area2 - inter
                    if inter / union > iou_threshold if union > 0 else False:
                        is_dup = True
                        break

            if not is_dup:
                keep.append(det)

        return keep

    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Phát hiện biển số trong ảnh
        """
        original_h, original_w = image.shape[:2]
        processed_img = self._preprocess_image(image.copy())

        detections = []

        # Thử model chuyên dụng trước
        self._load_model()

        if self.model is not None:
            try:
                results = self.model(
                    processed_img, verbose=False, conf=self.conf_threshold,
                    iou=0.45, max_det=5, half=False
                )
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            detections.append({
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'confidence': conf,
                                'class_id': cls_id,
                                'class_name': 'license_plate'
                            })
            except Exception as e:
                logger.warning(f"Model detection failed: {e}")

        # Fallback: OpenCV color detection
        if len(detections) == 0:
            detections = self._detect_by_color(processed_img)

        # Scale bbox về kích thước gốc
        if original_h != processed_img.shape[0] or original_w != processed_img.shape[1]:
            scale_x = original_w / processed_img.shape[1]
            scale_y = original_h / processed_img.shape[0]
            scaled = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                scaled.append({
                    'bbox': (int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)),
                    'confidence': det['confidence'],
                    'class_id': det['class_id'],
                    'class_name': det['class_name']
                })
            return scaled

        return detections

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[dict],
                       plate_texts: List[str] = None) -> np.ndarray:
        """Vẽ các detection lên ảnh"""
        result = image.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 0)  # Xanh lá

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label = det.get('class_name', 'plate')
            if plate_texts and i < len(plate_texts) and plate_texts[i]:
                label = plate_texts[i]

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result, (x1, y1 - lh - 10), (x1 + lw, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return result


# Singleton
_detector_instance = None


def get_detector(model_path: Optional[str] = None, conf_threshold: float = 0.5) -> PlateDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlateDetector(model_path, conf_threshold)
    return _detector_instance

    def detect_plate_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Phát hiện và trả về các vùng ảnh chứa biển số

        Args:
            image: Ảnh đầu vào

        Returns:
            List of cropped plate images
        """
        detections = self.detect(image)
        plates = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            plate = image[y1:y2, x1:x2]
            plates.append(plate)

        return plates

    def _detect_opencv(self, image: np.ndarray) -> List[dict]:
        """
        Fallback detection sử dụng OpenCV.
        Áp dụng các kỹ thuật xử lý ảnh để tìm vùng giống biển số.
        """
        detections = []

        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter để giảm noise
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)

        # Edge detection
        edged = cv2.Canny(blurred, 30, 200)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area và lấy top contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_contour = None
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            # Tìm contour có 4 đỉnh (hình chữ nhật)
            if len(approx) == 4:
                plate_contour = approx
                break

        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)

            # Kiểm tra tỷ lệ (biển số thường có tỷ lệ width/height = 2-6)
            aspect_ratio = w / h if h > 0 else 0
            if 1.5 < aspect_ratio < 6:
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 0.7,
                    'class_id': 0,
                    'class_name': 'license_plate'
                })

        return detections

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[dict],
                       plate_texts: List[str] = None) -> np.ndarray:
        """
        Vẽ các detection lên ảnh

        Args:
            image: Ảnh gốc
            detections: Kết quả từ detect()
            plate_texts: Danh sách text biển số tương ứng

        Returns:
            Ảnh có vẽ các bounding boxes
        """
        result = image.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']

            # Màu xanh lá
            color = (0, 255, 0)

            # Vẽ bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Vẽ text
            label = f"{det.get('class_name', 'plate')}: {conf:.2f}"
            if plate_texts and i < len(plate_texts):
                label = f"{plate_texts[i]} ({conf:.2f})"

            # Background cho text
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(result, (x1, y1 - label_h - 10),
                         (x1 + label_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result


# Singleton instance
_detector_instance = None


def get_detector(model_path: Optional[str] = None, conf_threshold: float = 0.5) -> PlateDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlateDetector(model_path, conf_threshold)
    return _detector_instance
