"""
Module OCR nhận diện ký tự từ biển số xe
"""

import cv2
import numpy as np
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class OCRRecognizer:
    """
    Nhận diện ký tự từ ảnh biển số - Tối ưu cho biển số VN
    """

    def __init__(self, languages: List[str] = None, use_gpu: bool = True):
        self.languages = languages or ['en']
        self.use_gpu = use_gpu and self._check_gpu()
        self.reader = None
        self._reader_loaded = False

    def _check_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _load_reader(self):
        if self._reader_loaded:
            return

        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
                download_enabled=True
            )
            self._reader_loaded = True
            logger.info(f"EasyOCR loaded: GPU={self.use_gpu}")
        except ImportError:
            logger.warning("EasyOCR not installed. Using fallback OCR.")
            self._reader_loaded = False

    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh biển số để cải thiện OCR"""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # Resize nếu ảnh quá nhỏ
        h, w = gray.shape
        if h < 80:
            scale = 80 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Tăng tương phản CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise nhẹ
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

        return denoised

    def recognize(self, plate_image: np.ndarray) -> List[dict]:
        """Nhận diện text từ ảnh biển số"""
        self._load_reader()

        # Preprocess trước
        processed = self._preprocess_plate(plate_image)

        if self.reader is None:
            return self._fallback_ocr(processed)

        try:
            # Convert to RGB
            if len(processed.shape) == 2:
                rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = processed

            results = self.reader.readtext(rgb_image)

            recognized = []
            for (bbox, text, conf) in results:
                if text and len(text.strip()) > 0:
                    recognized.append({
                        'text': text.strip(),
                        'confidence': float(conf),
                        'bbox': bbox
                    })

            return recognized

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return self._fallback_ocr(processed)

    def get_text(self, plate_image: np.ndarray) -> str:
        """Lấy text biển số đã format"""
        results = self.recognize(plate_image)

        if not results:
            return ""

        # Sort by x position
        sorted_results = sorted(results, key=lambda x: min(p[0] for p in x['bbox']))

        # Combine text
        full_text = ""
        for r in sorted_results:
            full_text += r['text'] + " "

        return self._format_vietnamese_plate(full_text.strip())

    def _format_vietnamese_plate(self, text: str) -> str:
        """Format text thành biển số VN chuẩn"""
        if not text:
            return ""

        # Clean text
        text = text.upper()
        text = text.replace(" ", "").replace("-", "").replace(".", "").replace(",", "")
        text = text.replace("O", "0").replace("I", "1").replace("L", "1")
        text = text.replace("S", "5").replace("B", "8").replace("Z", "2")

        # Vietnamese plate patterns
        patterns = [
            (r'^(\d{2})([A-Z])(\d{4,5})$', r'\1\2-\3'),  # 30A-12345
            (r'^(\d{2})([A-Z]{1,2})(\d{3})(\d{2})$', r'\1\2-\3.\4'),  # 30A1-123.45
            (r'^(\d{2})([A-Z])(\d{3})(\d{1,2})$', r'\1\2-\3\4'),  # 30A-1234
        ]

        for pattern, replacement in patterns:
            match = re.match(pattern, text)
            if match:
                return re.sub(pattern, replacement, text)

        return text

    def _fallback_ocr(self, plate_img: np.ndarray) -> List[dict]:
        """Fallback OCR với pytesseract"""
        try:
            import pytesseract

            if len(plate_img.shape) == 3:
                gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_img

            # Config cho biển số
            config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(gray, config=config)

            if text and text.strip():
                return [{
                    'text': text.strip(),
                    'confidence': 0.5,
                    'bbox': [(0, 0), (plate_img.shape[1], 0),
                            (plate_img.shape[1], plate_img.shape[0]), (0, plate_img.shape[0])]
                }]
        except Exception as e:
            logger.error(f"Fallback OCR error: {e}")

        return []


# Singleton
_ocr_instance = None


def get_ocr_recognizer(languages: List[str] = None, use_gpu: bool = True) -> OCRRecognizer:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCRRecognizer(languages, use_gpu)
    return _ocr_instance
