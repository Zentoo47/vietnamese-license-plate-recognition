"""
Module xử lý ảnh - Tiền xử lý ảnh đầu vào trước khi detect và OCR
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class ImageProcessor:
    """Xử lý ảnh: resize, enhance, convert sang grayscale, etc."""

    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Đọc ảnh từ file path"""
        try:
            img = cv2.imread(image_path)
            return img
        except Exception as e:
            print(f"Lỗi đọc ảnh: {e}")
            return None

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """Đọc ảnh từ bytes"""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Lỗi đọc ảnh từ bytes: {e}")
            return None

    @staticmethod
    def resize_image(img: np.ndarray, max_width: int = 800, max_height: int = 600) -> np.ndarray:
        """Resize ảnh giữ nguyên tỷ lệ"""
        height, width = img.shape[:2]

        # Tính tỷ lệ resize
        width_ratio = max_width / width
        height_ratio = max_height / height
        ratio = min(width_ratio, height_ratio, 1.0)

        if ratio < 1.0:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Chuyển ảnh sang grayscale"""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Tăng cường độ tương phản bằng CLAHE"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        enhanced = clahe.apply(img)
        return enhanced

    @staticmethod
    def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
        """Khử nhiễu ảnh"""
        return cv2.fastNlMeansDenoising(img, None, strength, 7, 21)

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """Làm sắc nét ảnh"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def adaptive_threshold(img: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
        """Adaptive threshold để tách biển số khỏi nền"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c
        )
        return thresh

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Xử lý nghiêng của biển số"""
        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        return rotated

    @staticmethod
    def remove_borders(img: np.ndarray, padding: int = 5) -> np.ndarray:
        """Loại bỏ viền đen không cần thiết"""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return img[y+padding:y+h-padding, x+padding:x+w-padding]

        return img

    @staticmethod
    def resize_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
        """Resize ảnh theo chiều cao cố định"""
        height = img.shape[0]
        ratio = target_height / height
        new_width = int(img.shape[1] * ratio)
        return cv2.resize(img, (new_width, target_height))

    @staticmethod
    def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh biển số cho OCR"""
        # Chuyển sang grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Resize nếu ảnh quá nhỏ
        if gray.shape[0] < 50:
            scale = 50 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Tăng cường tương phản
        enhanced = ImageProcessor.enhance_contrast(gray)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    @staticmethod
    def save_image(img: np.ndarray, output_path: str) -> bool:
        """Lưu ảnh ra file"""
        try:
            cv2.imwrite(output_path, img)
            return True
        except Exception as e:
            print(f"Lỗi lưu ảnh: {e}")
            return False

    @staticmethod
    def draw_boxes(img: np.ndarray, boxes: list, labels: list = None, scores: list = None) -> np.ndarray:
        """Vẽ bounding boxes lên ảnh"""
        result = img.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Màu xanh lá cho box
            color = (0, 255, 0)

            # Vẽ box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Vẽ label
            label = ""
            if labels and i < len(labels):
                label = labels[i]
                if scores and i < len(scores):
                    label += f" {scores[i]:.2f}"

            if label:
                cv2.putText(result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return result
