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

        h, w = gray.shape
        if h == 0 or w == 0:
            return gray

        target_height = 120
        if h < target_height:
            scale = target_height / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

        return denoised

    def _preprocess_variants(self, plate_img: np.ndarray) -> List[np.ndarray]:
        """Tạo nhiều biến thể ảnh để OCR ổn định hơn với ảnh mờ/tối/chói."""
        base = self._preprocess_plate(plate_img)
        if base.size == 0:
            return [base]

        variants = [base]
        blur = cv2.GaussianBlur(base, (3, 3), 0)
        variants.append(blur)

        binary = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 9
        )
        variants.append(binary)
        variants.append(cv2.bitwise_not(binary))

        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        unique = []
        seen_shapes = set()
        for image in variants:
            key = (image.shape, int(image.mean()))
            if key not in seen_shapes:
                unique.append(image)
                seen_shapes.add(key)
        return unique

    def recognize(self, plate_image: np.ndarray) -> List[dict]:
        """Nhận diện text từ ảnh biển số"""
        self._load_reader()

        variants = self._preprocess_variants(plate_image)

        if self.reader is None:
            return self._fallback_ocr(variants[0])

        try:
            best_results = []
            best_score = -1

            for processed in variants:
                if len(processed.shape) == 2:
                    rgb_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_image = processed

                results = self.reader.readtext(
                    rgb_image,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.',
                    decoder='beamsearch',
                    detail=1,
                    paragraph=False,
                    text_threshold=0.35,
                    low_text=0.25,
                    link_threshold=0.3
                )

                recognized = []
                for (bbox, text, conf) in results:
                    cleaned = re.sub(r'[^A-Za-z0-9]', '', text or '').upper()
                    if cleaned and len(cleaned) >= 2:
                        recognized.append({
                            'text': cleaned,
                            'confidence': float(conf),
                            'bbox': bbox
                        })

                candidate_text = ''.join(r['text'] for r in self._sort_ocr_results(recognized))
                score = self._score_plate_text(candidate_text)
                if recognized:
                    score += sum(r['confidence'] for r in recognized) / len(recognized)

                if score > best_score:
                    best_score = score
                    best_results = recognized

            return best_results

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return self._fallback_ocr(variants[0])

    def _sort_ocr_results(self, results: List[dict]) -> List[dict]:
        """Sắp xếp OCR theo dòng rồi theo vị trí ngang."""
        return sorted(
            results,
            key=lambda x: (
                min(p[1] for p in x['bbox']),
                min(p[0] for p in x['bbox'])
            )
        )

    def get_text(self, plate_image: np.ndarray) -> str:
        """Lấy text biển số đã format"""
        text, _confidence, _results = self.get_text_with_confidence(plate_image)
        return text

    def get_text_with_confidence(self, plate_image: np.ndarray) -> tuple:
        """Lấy text đã format cùng độ tin cậy OCR đã hiệu chỉnh."""
        results = self.recognize(plate_image)

        if not results:
            return "", 0.0, []

        sorted_results = self._sort_ocr_results(results)

        full_text = ""
        for r in sorted_results:
            full_text += r['text'] + " "

        formatted = self._format_vietnamese_plate(full_text.strip())
        ocr_confidence = sum(r['confidence'] for r in sorted_results) / len(sorted_results)
        confidence = self._calibrated_confidence(formatted, ocr_confidence)
        return formatted, confidence, sorted_results

    def _calibrated_confidence(self, text: str, ocr_confidence: float) -> float:
        """Ước lượng confidence thực tế, phạt mạnh nếu sai format biển số VN."""
        clean = re.sub(r'[^A-Z0-9]', '', (text or '').upper())
        if not clean:
            return 0.0

        format_score = self._score_plate_text(clean)
        format_score = min(format_score / 4.0, 1.0)
        length_penalty = 1.0 if 7 <= len(clean) <= 9 else 0.55
        province_bonus = 1.0 if len(clean) >= 2 and clean[:2].isdigit() else 0.65
        series_bonus = 1.0 if len(clean) >= 3 and clean[2].isalpha() else 0.7

        calibrated = ocr_confidence * (0.45 + 0.55 * format_score)
        calibrated *= length_penalty * province_bonus * series_bonus
        return float(max(0.0, min(calibrated, 0.99)))

    def _score_plate_text(self, text: str) -> float:
        """Chấm điểm ứng viên OCR theo format biển số Việt Nam."""
        raw = re.sub(r'[^A-Z0-9]', '', (text or '').upper())
        if not raw:
            return 0

        formatted = self._format_vietnamese_plate(raw)
        clean = re.sub(r'[^A-Z0-9]', '', formatted)
        score = min(len(clean), 9) / 9

        if re.match(r'^\d{2}[A-Z]{1,2}\d{4,5}$', clean):
            score += 2.0
        if 7 <= len(clean) <= 9:
            score += 0.5
        if clean[:2].isdigit():
            score += 0.5
        if len(clean) >= 3 and clean[2].isalpha():
            score += 0.5
        return score

    def _format_vietnamese_plate(self, text: str) -> str:
        """Format text thành biển số VN chuẩn"""
        if not text:
            return ""

        # Clean text. Do not blindly convert letters such as B/S/Z to digits,
        # because Vietnamese plates contain real letters in the series section.
        raw = re.sub(r'[^A-Z0-9]', '', text.upper())
        if not raw:
            return ""

        digit_map = str.maketrans({'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'T': '7'})
        letter_map = str.maketrans({'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z', '4': 'A'})

        def normalize_digit(value: str) -> str:
            return value.translate(digit_map) if value.isalpha() else value

        def normalize_letter(value: str) -> str:
            return value.translate(letter_map) if value.isdigit() else value

        chars = list(raw)
        # First two characters are province digits.
        for i in range(min(2, len(chars))):
            chars[i] = normalize_digit(chars[i])

        # Series characters directly after province are letters, optionally followed by one digit.
        if len(chars) >= 3:
            chars[2] = normalize_letter(chars[2])
        if len(chars) >= 4 and chars[3].isalpha():
            chars[3] = normalize_letter(chars[3])

        # Remaining characters are serial digits.
        start_digits = 4 if len(chars) >= 4 and chars[3].isalpha() else 3
        for i in range(start_digits, len(chars)):
            chars[i] = normalize_digit(chars[i])

        text = ''.join(chars)

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
