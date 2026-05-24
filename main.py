"""
Hệ thống nhận diện biển số xe - Main Application
Intelligent Traffic System (ITS)
"""

import os
import sys
import argparse
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api import router
from app.database import init_db
from app.models import PlateDetector, OCRRecognizer
from app.utils import ImageProcessor, PlateUtils


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler thay thế on_event"""
    logger.info("Khởi động hệ thống nhận diện biển số xe...")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo database: {e}")

    # Create directories
    for dir_name in ["uploads", "results", "data"]:
        dir_path = os.path.join(BASE_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=True)

    # Pre-load models (warmup)
    try:
        from app.models.detector import get_detector
        from app.models.ocr_recognizer import get_ocr_recognizer

        detector = get_detector()
        ocr = get_ocr_recognizer(languages=['en'])

        # Warmup models với dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        detector.detect(dummy_img)
        logger.info("Models warmed up successfully")
    except Exception as e:
        logger.warning(f"Không thể warmup models: {e}")

    logger.info("Hệ thống sẵn sàng!")

    yield  # App đang chạy

    logger.info("Tắt hệ thống...")


# Create FastAPI app
app = FastAPI(
    title="ITS - Hệ thống nhận diện biển số xe",
    description="Ứng dụng nhận diện biển số xe tự động sử dụng Deep Learning và OCR",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Include API routes
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


class LicensePlateRecognitionPipeline:
    """
    Pipeline chính cho nhận diện biển số xe.
    Có thể sử dụng độc lập hoặc qua API.
    """

    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Khởi tạo pipeline

        Args:
            model_path: Đường dẫn model YOLO (optional)
            use_gpu: Có sử dụng GPU không
        """
        self.detector = PlateDetector(model_path)
        self.ocr = OCRRecognizer(use_gpu=use_gpu)
        self.image_processor = ImageProcessor()

        logger.info("Pipeline initialized")

    def process_image(self, image_path: str, save_result: bool = True) -> dict:
        """
        Xử lý 1 ảnh

        Args:
            image_path: Đường dẫn ảnh
            save_result: Có lưu ảnh kết quả không

        Returns:
            Dict chứa kết quả nhận diện
        """
        import time
        start_time = time.time()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"success": False, "error": "Không thể đọc ảnh"}

        # Detect plates
        detections = self.detector.detect(image)

        results = []
        for det in detections:
            bbox = det['bbox']

            # Crop plate
            plate_img = PlateUtils.crop_plate(image, bbox)

            # OCR
            plate_text = self.ocr.get_text(plate_img)
            ocr_results = self.ocr.recognize(plate_img)

            # Calculate confidence
            conf = det['confidence']
            if ocr_results:
                avg_ocr_conf = sum(r['confidence'] for r in ocr_results) / len(ocr_results)
                conf = (conf + avg_ocr_conf) / 2

            # Get vehicle type
            vehicle_type = PlateUtils.get_plate_type(plate_text)

            result = {
                'plate_text': plate_text or "UNKNOWN",
                'confidence': float(conf),
                'vehicle_type': vehicle_type,
                'bbox': bbox,
                'ocr_results': ocr_results,
                'processing_time': time.time() - start_time
            }
            results.append(result)

        # Draw results
        if save_result and results:
            plate_texts = [r['plate_text'] for r in results]
            result_img = PlateDetector.draw_detections(image, detections, plate_texts)

            # Save
            output_path = image_path.rsplit('.', 1)[0] + '_result.jpg'
            cv2.imwrite(output_path, result_img)
            logger.info(f"Kết quả lưu tại: {output_path}")

        return {
            "success": True,
            "image_path": image_path,
            "total_plates": len(results),
            "results": results,
            "total_time": time.time() - start_time
        }

    def process_video(self, video_path: str, output_path: str = None,
                     fps_display: int = 1) -> dict:
        """
        Xử lý video - nhận diện từng frame

        Args:
            video_path: Đường dẫn video đầu vào
            output_path: Đường dẫn video đầu ra (optional)
            fps_display: Số fps hiển thị kết quả

        Returns:
            Dict chứa kết quả
        """
        import time

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"success": False, "error": "Không thể mở video"}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        all_results = []
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect every fps_display frame
            if frame_count % fps_display == 0:
                detections = self.detector.detect(frame)
                plate_texts = []

                for det in detections:
                    bbox = det['bbox']
                    plate_img = PlateUtils.crop_plate(frame, bbox)
                    plate_text = self.ocr.get_text(plate_img)
                    plate_texts.append(plate_text)

                    all_results.append({
                        'frame': frame_count,
                        'plate_text': plate_text,
                        'bbox': bbox,
                        'confidence': det['confidence']
                    })

                # Draw
                result_frame = PlateDetector.draw_detections(frame, detections, plate_texts)
            else:
                result_frame = frame

            if writer:
                writer.write(result_frame)

            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        cap.release()
        if writer:
            writer.release()

        return {
            "success": True,
            "total_frames": frame_count,
            "detections": len(all_results),
            "results": all_results,
            "processing_time": time.time() - start_time,
            "output_path": output_path
        }


def run_cli(image_path: str = None):
    """Chạy từ command line"""
    if image_path:
        print(f"\n{'='*50}")
        print("HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE")
        print(f"{'='*50}\n")

        pipeline = LicensePlateRecognitionPipeline()
        result = pipeline.process_image(image_path)

        if result['success']:
            print(f"\nẢnh: {result['image_path']}")
            print(f"Tổng biển số: {result['total_plates']}")
            print(f"Thời gian xử lý: {result['total_time']*1000:.0f}ms")
            print(f"\nKết quả:")
            for i, r in enumerate(result['results'], 1):
                print(f"  {i}. Biển số: {r['plate_text']}")
                print(f"     Loại xe: {r['vehicle_type']}")
                print(f"     Độ chính xác: {r['confidence']*100:.1f}%")
        else:
            print(f"Lỗi: {result.get('error', 'Unknown error')}")
    else:
        print("Sử dụng: python main.py --image <đường_dẫn_ảnh>")
        print("Hoặc chạy: uvicorn main:app --reload")


def run_api():
    """Chạy API server"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ thống nhận diện biển số xe")
    parser.add_argument("--image", type=str, help="Đường dẫn ảnh để xử lý")
    parser.add_argument("--video", type=str, help="Đường dẫn video để xử lý")
    parser.add_argument("--api", action="store_true", help="Chạy API server")
    parser.add_argument("--gpu", action="store_true", default=True, help="Sử dụng GPU")

    args = parser.parse_args()

    if args.api:
        print("Khởi động API server...")
        run_api()
    elif args.image:
        run_cli(args.image)
    elif args.video:
        pipeline = LicensePlateRecognitionPipeline()
        output = args.video.rsplit('.', 1)[0] + '_output.mp4'
        result = pipeline.process_video(args.video, output)
        print(f"\nKết quả: {result['detections']} biển số được nhận diện")
        print(f"Video lưu tại: {output}")
    else:
        print("Khởi động API server mặc định...")
        run_api()
