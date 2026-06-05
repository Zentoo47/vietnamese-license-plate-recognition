"""
FastAPI routes cho API endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import uuid
import base64
from io import BytesIO
import asyncio
import logging

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class DetectionResult(BaseModel):
    plate_text: str
    confidence: float
    vehicle_type: str
    bbox: List[int]
    processing_time: float


class DetectionResponse(BaseModel):
    success: bool
    results: List[DetectionResult]
    total_plates: int
    total_time: float
    image_result: Optional[str] = None


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.post("/detect", response_model=DetectionResponse)
async def detect_plate(file: UploadFile = File(...)):
    """
    Nhận diện biển số từ ảnh upload
    """
    import time
    import cv2

    start_time = time.time()

    allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="File type not supported")

    try:
        contents = await file.read()

        # Load image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Cannot read image")

        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _process_detection, image)

        # Encode result image
        if result['image_result'] is not None:
            _, buffer = cv2.imencode('.jpg', result['image_result'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result['image_result'] = img_base64

        return DetectionResponse(
            success=True,
            results=[DetectionResult(**r) for r in result['results']],
            total_plates=result['total_plates'],
            total_time=time.time() - start_time,
            image_result=result.get('image_result')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _process_detection(image: np.ndarray) -> dict:
    """Xử lý detection (chạy trong thread pool)"""
    import time
    import cv2

    from app.models import PlateDetector, OCRRecognizer
    from app.models.detector import get_detector
    from app.models.ocr_recognizer import get_ocr_recognizer
    from app.utils import ImageProcessor, PlateUtils

    start_time = time.time()

    # Initialize models
    detector = get_detector(conf_threshold=0.5)
    ocr = get_ocr_recognizer(languages=['en'])

    # Detect plates
    detections = detector.detect(image)

    results = []
    plate_texts = []

    for det in detections:
        bbox = det['bbox']
        plate_img = PlateUtils.crop_plate(image, bbox)
        plate_text, conf, ocr_results = ocr.get_text_with_confidence(plate_img)

        vehicle_type = PlateUtils.get_plate_type(plate_text)

        results.append({
            'plate_text': plate_text or "UNKNOWN",
            'confidence': float(conf),
            'vehicle_type': vehicle_type,
            'bbox': bbox,
            'processing_time': time.time() - start_time
        })
        plate_texts.append(plate_text or "UNKNOWN")

    # Draw results
    result_img = PlateDetector.draw_detections(image, detections, plate_texts)

    # Save result
    result_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
    result_path = os.path.join(result_dir, result_filename)
    cv2.imwrite(result_path, result_img)

    # Save to database
    try:
        from app.database import DetectionRecord, get_session
        session = get_session()
        for r in results:
            record = DetectionRecord(
                plate_text=r['plate_text'],
                confidence=r['confidence'],
                vehicle_type=r['vehicle_type'],
                processing_time=r['processing_time'],
                image_path=file.filename if 'file' in dir() else 'upload',
                result_image_path=result_path,
                image_width=image.shape[1],
                image_height=image.shape[0],
                bbox_x1=r['bbox'][0],
                bbox_y1=r['bbox'][1],
                bbox_x2=r['bbox'][2],
                bbox_y2=r['bbox'][3]
            )
            session.add(record)
        session.commit()
        session.close()
    except Exception as db_error:
        logger.error(f"Database error: {db_error}")

    return {
        'results': results,
        'total_plates': len(results),
        'image_result': result_img
    }



@router.post("/detect-video")
async def detect_video(file: UploadFile = File(...)):
    """Nhận diện biển số từ video upload"""
    import time
    import cv2

    start_time = time.time()
    allowed_types = {
        'video/mp4',
        'video/mpeg',
        'video/quicktime',
        'video/x-msvideo',
        'video/x-matroska',
        'application/octet-stream',
    }
    allowed_exts = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    ext = os.path.splitext(file.filename or '')[1].lower()

    if file.content_type not in allowed_types and ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="File video không được hỗ trợ")

    upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    input_filename = f"video_{uuid.uuid4().hex[:8]}{ext or '.mp4'}"
    input_path = os.path.join(upload_dir, input_filename)

    try:
        with open(input_path, 'wb') as video_file:
            while chunk := await file.read(1024 * 1024):
                video_file.write(chunk)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _process_video_detection, input_path, result_dir)
        result['total_time'] = time.time() - start_time
        result['success'] = True
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{filename}")
async def get_result_file(filename: str):
    """Trả file kết quả đã xử lý"""
    safe_name = os.path.basename(filename)
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results')
    file_path = os.path.join(result_dir, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả")
    media_type = "video/mp4" if safe_name.lower().endswith(".mp4") else None
    return FileResponse(file_path, media_type=media_type)

def _open_video_writer(output_path: str, fps: float, size: tuple):
    """Create a browser-friendly MP4 writer with codec fallback."""
    import cv2

    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    return None

def _process_video_detection(input_path: str, result_dir: str) -> dict:
    """Xử lý video và vẽ kết quả nhận diện lên từng frame"""
    import time
    import cv2

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Không thể đọc video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if fps <= 1 or fps > 120:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_width = width - (width % 2)
    output_height = height - (height % 2)

    output_filename = f"result_video_{uuid.uuid4().hex[:8]}.mp4"
    output_path = os.path.join(result_dir, output_filename)
    writer = _open_video_writer(output_path, fps, (output_width, output_height))

    if writer is None:
        cap.release()
        raise ValueError("Không thể tạo video kết quả")

    from app.models import PlateDetector
    from app.models.detector import get_detector
    from app.models.ocr_recognizer import get_ocr_recognizer
    from app.utils import PlateUtils

    detector = get_detector(conf_threshold=0.5)
    ocr = get_ocr_recognizer(languages=['en'])

    frame_index = 0
    detections_count = 0
    best_results = []
    process_every = max(1, int(round(fps / 3)))
    if total_frames > 600:
        process_every = max(process_every, total_frames // 600)

    last_detections = []
    last_plate_texts = []
    started_at = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % process_every == 0:
            last_detections = detector.detect(frame)
            last_plate_texts = []

            for det in last_detections:
                bbox = det['bbox']
                plate_img = PlateUtils.crop_plate(frame, bbox)
                plate_text, confidence, ocr_results = ocr.get_text_with_confidence(plate_img)
                plate_text = plate_text or "UNKNOWN"

                result = {
                    'plate_text': plate_text,
                    'confidence': float(confidence),
                    'vehicle_type': PlateUtils.get_plate_type(plate_text),
                    'bbox': bbox,
                    'frame': frame_index,
                    'processing_time': time.time() - started_at,
                }
                best_results.append(result)
                last_plate_texts.append(plate_text)
                detections_count += 1

        output_frame = PlateDetector.draw_detections(frame, last_detections, last_plate_texts)
        if output_frame.shape[1] != output_width or output_frame.shape[0] != output_height:
            output_frame = cv2.resize(output_frame, (output_width, output_height))
        writer.write(output_frame)
        frame_index += 1

    cap.release()
    writer.release()

    best_results.sort(key=lambda item: item['confidence'], reverse=True)
    unique_results = []
    seen = set()
    for item in best_results:
        key = item['plate_text'].strip().upper()
        if not key or key == "UNKNOWN":
            continue
        if key not in seen:
            unique_results.append(item)
            seen.add(key)
        if len(unique_results) >= 10:
            break

    return {
        'results': unique_results,
        'total_plates': detections_count,
        'total_frames': frame_index,
        'processed_every_n_frames': process_every,
        'video_result': f"/api/results/{output_filename}",
    }

@router.get("/history", response_model=List[dict])
async def get_history(limit: int = 50, offset: int = 0):
    """Lấy lịch sử nhận diện"""
    try:
        from app.database import get_session, DetectionRecord
        session = get_session()
        records = session.query(DetectionRecord)\
                        .order_by(DetectionRecord.created_at.desc())\
                        .offset(offset)\
                        .limit(limit)\
                        .all()
        result = [r.to_dict() for r in records]
        session.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Lấy thống kê"""
    try:
        from app.database import get_session, DetectionRecord
        from sqlalchemy import func

        session = get_session()
        total = session.query(DetectionRecord).count()
        avg_confidence = session.query(func.avg(DetectionRecord.confidence)).scalar() or 0
        avg_processing_time = session.query(func.avg(DetectionRecord.processing_time)).scalar() or 0

        vehicle_stats = session.query(
            DetectionRecord.vehicle_type,
            func.count(DetectionRecord.id)
        ).group_by(DetectionRecord.vehicle_type).all()

        session.close()

        return {
            'total_detections': total,
            'avg_confidence': float(avg_confidence),
            'avg_processing_time': float(avg_processing_time),
            'by_vehicle_type': {v: c for v, c in vehicle_stats}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def clear_history():
    """Xóa lịch sử"""
    try:
        from app.database import get_session, DetectionRecord
        session = get_session()
        session.query(DetectionRecord).delete()
        session.commit()
        session.close()
        return {"message": "History cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
