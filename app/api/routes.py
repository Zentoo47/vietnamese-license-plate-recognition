"""
FastAPI routes cho API endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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
        plate_text = ocr.get_text(plate_img)
        ocr_results = ocr.recognize(plate_img)

        conf = det['confidence']
        if ocr_results:
            avg_ocr_conf = sum(r['confidence'] for r in ocr_results) / len(ocr_results)
            conf = (conf + avg_ocr_conf) / 2

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
