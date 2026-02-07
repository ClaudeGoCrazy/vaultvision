import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.core.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=gen_uuid)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    progress_percent = Column(Float, default=0.0)
    current_step = Column(String, nullable=True)
    duration_sec = Column(Float, nullable=True)
    upload_time = Column(DateTime, default=utcnow)
    completed_time = Column(DateTime, nullable=True)
    total_detections = Column(Integer, default=0)
    unique_persons = Column(Integer, default=0)
    unique_vehicles = Column(Integer, default=0)
    total_frames = Column(Integer, default=0)
    fps_processed = Column(Float, default=0.0)
    processing_time_sec = Column(Float, default=0.0)
    thumbnail_path = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="video", cascade="all, delete-orphan")
    heatmap = relationship("Heatmap", back_populates="video", uselist=False, cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(String, primary_key=True, default=gen_uuid)
    video_id = Column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    detection_id = Column(String, nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    track_id = Column(Integer, nullable=True)

    video = relationship("Video", back_populates="detections")


class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True, default=gen_uuid)
    video_id = Column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    event_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    track_id = Column(Integer, nullable=True)
    start_time_sec = Column(Float, nullable=False)
    end_time_sec = Column(Float, nullable=True)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    metadata_json = Column(JSON, default=dict)

    video = relationship("Video", back_populates="events")


class Heatmap(Base):
    __tablename__ = "heatmaps"

    id = Column(String, primary_key=True, default=gen_uuid)
    video_id = Column(String, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    grid_json = Column(JSON, nullable=False)
    video_width = Column(Integer, nullable=False)
    video_height = Column(Integer, nullable=False)

    video = relationship("Video", back_populates="heatmap")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, default=gen_uuid)
    key = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    is_active = Column(Integer, default=1)
