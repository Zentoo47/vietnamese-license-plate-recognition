"""
License plate detection module using YOLO when available and OpenCV fallback.
"""

import logging
import os
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Detect license plate regions.

    Priority:
    1. Custom/fine-tuned YOLO model if available.
    2. OpenCV color/contour fallback for Vietnamese plates.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        max_image_size: int = 800,
    ):
        self.conf_threshold = conf_threshold
        self.max_image_size = max_image_size
        self.model_path = model_path or self._get_best_model_path()
        self.model = None
        self._model_loaded = False
        self._model_type = None

    def _get_best_model_path(self) -> Optional[str]:
        """Return the best available local YOLO model path."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        models_dir = os.path.join(base_dir, "src", "trained_models")
        root_models_dir = base_dir

        candidates = [
            os.path.join(models_dir, "license_plate_yolov8n.pt"),
            os.path.join(models_dir, "vietnamese_license_plate", "weights", "best.pt"),
            os.path.join(models_dir, "anpr_demo.pt"),
            os.path.join(root_models_dir, "yolov8m.pt"),
            os.path.join(root_models_dir, "yolov8n.pt"),
        ]

        for path in candidates:
            if os.path.exists(path):
                logger.info("Found detector model: %s", path)
                return path

        return None

    def _load_model(self) -> None:
        """Lazy-load YOLO model, falling back to OpenCV on failure."""
        if self._model_loaded:
            return

        try:
            from ultralytics import YOLO
            import torch

            if self.model_path and os.path.exists(self.model_path):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading YOLO detector on %s: %s", device, self.model_path)
                self.model = YOLO(self.model_path)
                self.model.to(device)
                self._model_type = "yolo"
            else:
                logger.info("No detector model found; using OpenCV fallback.")
                self._model_type = "opencv"
        except Exception as exc:
            logger.warning("Could not load YOLO detector; using OpenCV fallback: %s", exc)
            self.model = None
            self._model_type = "opencv"

        self._model_loaded = True

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Resize very large images to keep detection fast."""
        height, width = image.shape[:2]
        if max(height, width) <= self.max_image_size:
            return image

        scale = self.max_image_size / max(height, width)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _detect_by_color(self, image: np.ndarray) -> List[dict]:
        """Detect Vietnamese-style white/blue plates using OpenCV heuristics."""
        detections = []
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        masks = []

        blue_mask = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
        masks.append((blue_mask, "blue_plate", 0.85))

        white_mask = cv2.inRange(gray, 190, 255)
        masks.append((white_mask, "white_plate", 0.75))

        yellow_mask = cv2.inRange(hsv, np.array([15, 50, 80]), np.array([40, 255, 255]))
        masks.append((yellow_mask, "yellow_plate", 0.75))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        max_area = width * height * 0.25

        for mask, class_name, confidence in masks:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, box_width, box_height = cv2.boundingRect(contour)
                if box_height == 0:
                    continue

                aspect_ratio = box_width / box_height
                area = box_width * box_height
                if 1.4 <= aspect_ratio <= 6.5 and 400 <= area <= max_area:
                    detections.append({
                        "bbox": (x, y, x + box_width, y + box_height),
                        "confidence": confidence,
                        "class_id": 0,
                        "class_name": class_name,
                    })

        return self._simple_nms(detections)

    def _detect_opencv(self, image: np.ndarray) -> List[dict]:
        """Detect plate-like rectangular regions using edges and color fallback."""
        detections = self._detect_by_color(image)
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blurred, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_height == 0:
                continue

            aspect_ratio = box_width / box_height
            area = box_width * box_height
            if 1.5 <= aspect_ratio <= 6.5 and 500 <= area <= width * height * 0.2:
                detections.append({
                    "bbox": (x, y, x + box_width, y + box_height),
                    "confidence": 0.65,
                    "class_id": 0,
                    "class_name": "license_plate",
                })

        return self._simple_nms(detections)

    def _simple_nms(self, detections: List[dict], iou_threshold: float = 0.3) -> List[dict]:
        """Apply non-maximum suppression to detection boxes."""
        if not detections:
            return []

        boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
        scores = np.array([det["confidence"] for det in detections], dtype=np.float32)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            current = order[0]
            keep.append(current)

            if order.size == 1:
                break

            x1 = np.maximum(boxes[current, 0], boxes[order[1:], 0])
            y1 = np.maximum(boxes[current, 1], boxes[order[1:], 1])
            x2 = np.minimum(boxes[current, 2], boxes[order[1:], 2])
            y2 = np.minimum(boxes[current, 3], boxes[order[1:], 3])

            inter_width = np.maximum(0, x2 - x1)
            inter_height = np.maximum(0, y2 - y1)
            intersection = inter_width * inter_height

            current_area = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
            other_areas = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
            union = current_area + other_areas - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

            order = order[1:][iou < iou_threshold]

        return [detections[index] for index in keep]

    def detect(self, image: np.ndarray) -> List[dict]:
        """Detect license plates and return bbox/confidence dictionaries."""
        self._load_model()
        original_height, original_width = image.shape[:2]
        processed = self._preprocess_image(image)
        proc_height, proc_width = processed.shape[:2]

        if self.model is None:
            detections = self._detect_opencv(processed)
        else:
            detections = []
            try:
                yolo_results = self.model(processed, conf=self.conf_threshold, verbose=False)
                for result in yolo_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                        class_name = self.model.names.get(class_id, "license_plate") if hasattr(self.model, "names") else "license_plate"
                        detections.append({
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                        })
            except Exception as exc:
                logger.warning("YOLO detection failed; using OpenCV fallback: %s", exc)
                detections = self._detect_opencv(processed)

        if processed.shape[:2] != image.shape[:2]:
            scale_x = original_width / proc_width
            scale_y = original_height / proc_height
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )

        return self._simple_nms(detections)

    def detect_plate_regions(self, image: np.ndarray) -> List[np.ndarray]:
        """Return cropped plate regions from an image."""
        crops = []
        for detection in self.detect(image):
            x1, y1, x2, y2 = detection["bbox"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
        return crops

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[dict], plate_texts: Optional[List[str]] = None) -> np.ndarray:
        """Draw detection boxes and optional recognized plate text."""
        result = image.copy()

        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection.get("confidence", 0.0)
            color = (0, 255, 0)

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = f"{detection.get('class_name', 'plate')}: {confidence:.2f}"
            if plate_texts and index < len(plate_texts) and plate_texts[index]:
                label = f"{plate_texts[index]} ({confidence:.2f})"

            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(y1, label_height + 10)
            cv2.rectangle(result, (x1, text_y - label_height - 10), (x1 + label_width + 6, text_y), color, -1)
            cv2.putText(result, label, (x1 + 3, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return result


_detector_instance = None


def get_detector(model_path: Optional[str] = None, conf_threshold: float = 0.5) -> PlateDetector:
    """Return a singleton detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PlateDetector(model_path=model_path, conf_threshold=conf_threshold)
    return _detector_instance
