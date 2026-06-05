"""
Tiện ích xử lý biển số xe
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Optional


class PlateUtils:
    """Các hàm tiện ích cho xử lý biển số xe"""

    # Regex pattern cho biển số xe Việt Nam
    VIETNAMESE_PLATE_PATTERNS = [
        # Biển số 2 dòng (xe máy): XX-XXXXX hoặc XX-XXXX
        r'[0-9]{2}[A-Z]-[0-9]{4,5}',
        # Biển số 1 dòng (ô tô): XXA-XXX.XX hoặc XX-XXX.XX
        r'[0-9]{2}[A-Z]{1,2}-[0-9]{3}[\.,][0-9]{2}',
        # Biển số cũ: XX-XXXXX
        r'[0-9]{2}-[0-9]{5}',
    ]

    @staticmethod
    def crop_plate(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Cắt vùng biển số từ ảnh gốc"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = img.shape[:2]
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        pad_x = max(6, int(box_w * 0.08))
        pad_y = max(6, int(box_h * 0.18))

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return img[y1:y2, x1:x2]

    @staticmethod
    def preprocess_plate_image(plate_img: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh biển số để cải thiện OCR"""
        # Chuyển sang grayscale
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # Resize nếu quá nhỏ
        h, w = gray.shape
        if h < 60 or w < 120:
            scale = max(120 / w, 60 / h)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Tăng tương phản với CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 8
        )

        # Morphological operations để loại bỏ noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return processed

    @staticmethod
    def is_valid_plate_format(text: str) -> bool:
        """Kiểm tra xem text có format biển số VN hợp lệ không"""
        if not text:
            return False

        # Clean text
        text = text.upper().replace(" ", "").replace(".", ",").replace("-", "")

        # Check patterns
        for pattern in PlateUtils.VIETNAMESE_PLATE_PATTERNS:
            if re.match(pattern.replace("-", ""), text):
                return True

        # Format xe máy: 2 số + 1 chữ + 4-5 số
        if re.match(r'^[0-9]{2}[A-Z][0-9]{4,5}$', text):
            return True

        # Format ô tô: 2 số + 1-2 chữ + 3 số + 2 số
        if re.match(r'^[0-9]{2}[A-Z]{1,2}[0-9]{5}$', text):
            return True

        return False

    @staticmethod
    def format_plate_text(text: str) -> str:
        """Format lại text biển số theo format chuẩn"""
        # Clean
        text = text.upper().replace(" ", "").replace("O", "0")

        # Try to detect format and reformat
        # Xe máy: 30A-12345
        match = re.match(r'^([0-9]{2})([A-Z])([0-9]{4,5})$', text)
        if match:
            return f"{match.group(1)}{match.group(2)}-{match.group(3)}"

        # Ô tô: 30A-123.45
        match = re.match(r'^([0-9]{2})([A-Z]{1,2})([0-9]{3})([0-9]{2})$', text)
        if match:
            return f"{match.group(1)}{match.group(2)}-{match.group(3)}.{match.group(4)}"

        return text

    @staticmethod
    def extract_characters(img: np.ndarray) -> List[np.ndarray]:
        """Tách các ký tự từ ảnh biển số"""
        chars = []

        # Binary threshold
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by x coordinate
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out small/noise contours
            if w > 5 and h > 10 and h < img.shape[0] * 0.9:
                char_img = img[y:y+h, x:x+w]
                chars.append(char_img)

        return chars

    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> float:
        """Tính Intersection over Union của 2 boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def non_max_suppression(boxes: List[Tuple[int, int, int, int]],
                           scores: List[float],
                           iou_threshold: float = 0.5) -> List[int]:
        """Non-Maximum Suppression để loại bỏ overlapping boxes"""
        if len(boxes) == 0:
            return []

        boxes_array = np.array(boxes)
        scores_array = np.array(scores)

        # Sort by scores
        indices = np.argsort(scores_array)[::-1]

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            ious = [PlateUtils.calculate_iou(boxes_array[current], boxes_array[i])
                   for i in indices[1:]]

            # Keep boxes with IoU < threshold
            indices = indices[1:][np.array(ious) < iou_threshold]

        return keep

    @staticmethod
    def get_plate_type(text: str) -> str:
        """Xác định loại phương tiện từ biển số"""
        text = text.upper().replace(" ", "").replace("-", "")

        # Biển số xe máy: 2 số + 1 chữ + 4-5 số (không có dấu chấm)
        if re.match(r'^[0-9]{2}[A-Z][0-9]{4,5}$', text):
            return "Xe máy"

        # Biển số ô tô: có 5 số, có thể có dấu chấm ngăn cách
        if re.match(r'^[0-9]{2}[A-Z]{1,2}[0-9]{5}$', text):
            return "Ô tô"

        # Biển số xe tải/nặng
        if text.startswith(('D', 'H', 'K', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X')):
            return "Xe tải"

        return "Không xác định"

    @staticmethod
    def visualize_plate(img: np.ndarray, plate_text: str,
                       bbox: Tuple[int, int, int, int],
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Vẽ biển số lên ảnh với text"""
        result = img.copy()

        x1, y1, x2, y2 = map(int, bbox)

        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # Draw background for text
        text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (x1, y1 - text_size[1] - 10),
                      (x1 + text_size[0] + 10, y1), color, -1)

        # Draw text
        cv2.putText(result, plate_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return result
