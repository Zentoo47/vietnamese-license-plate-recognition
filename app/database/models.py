"""
Database models cho lưu trữ kết quả nhận diện
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class DetectionRecord(Base):
    """Model lưu trữ kết quả nhận diện biển số"""

    __tablename__ = 'detection_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_text = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    image_path = Column(String(500), nullable=True)
    result_image_path = Column(String(500), nullable=True)
    vehicle_type = Column(String(50), nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    bbox_x1 = Column(Integer, nullable=True)
    bbox_y1 = Column(Integer, nullable=True)
    bbox_x2 = Column(Integer, nullable=True)
    bbox_y2 = Column(Integer, nullable=True)

    # OCR details
    ocr_confidence = Column(Float, nullable=True)
    raw_ocr_text = Column(Text, nullable=True)

    def __repr__(self):
        return f"<DetectionRecord(id={self.id}, plate='{self.plate_text}', conf={self.confidence:.2f})>"

    def to_dict(self):
        return {
            'id': self.id,
            'plate_text': self.plate_text,
            'confidence': self.confidence,
            'vehicle_type': self.vehicle_type,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'image_path': self.image_path,
        }


# Database setup
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'detections.db')


def init_db(db_path: str = None):
    """Khởi tạo database"""
    if db_path is None:
        db_path = DB_PATH

    # Create data directory if not exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    engine = create_engine(f'sqlite:///{db_path}')

    Base.metadata.create_all(engine)

    return engine


def get_session(db_path: str = None):
    """Get database session"""
    if db_path is None:
        db_path = DB_PATH

    engine = create_engine(f'sqlite:///{db_path}')
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    return Session()
