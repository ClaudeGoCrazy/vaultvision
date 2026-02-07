"""
VaultVision Shared Data Contracts
==================================
ALL THREE TERMINALS READ THIS FILE.
Do NOT modify without coordinating across all terminals.

Vault Sync AI LLC - VaultVision Product
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


# ============================================================
# ENUMS
# ============================================================

class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DetectionClass(str, Enum):
    PERSON = "person"
    VEHICLE = "vehicle"
    BICYCLE = "bicycle"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"
    CAR = "car"
    DOG = "dog"
    CAT = "cat"
    BACKPACK = "backpack"
    HANDBAG = "handbag"
    SUITCASE = "suitcase"
    CELL_PHONE = "cell_phone"
    LICENSE_PLATE = "license_plate"
    OTHER = "other"


class EventType(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    LOITERING = "loitering"
    ZONE_INTRUSION = "zone_intrusion"
    CROWD_THRESHOLD = "crowd_threshold"
    OBJECT_LEFT = "object_left"
    ANOMALY = "anomaly"


# ============================================================
# BOUNDING BOX & DETECTION (ML -> Backend -> Frontend)
# ============================================================

class BoundingBox(BaseModel):
    """Pixel coordinates of detection bounding box."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class Detection(BaseModel):
    """Single object detection in a single frame."""
    detection_id: str = Field(..., description="Unique detection ID")
    frame_number: int = Field(..., description="Frame index in video")
    timestamp_sec: float = Field(..., description="Timestamp in seconds")
    class_name: DetectionClass = Field(..., description="Detected object class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    track_id: Optional[int] = Field(None, description="Persistent tracking ID across frames")


# ============================================================
# EVENTS (ML generates, Backend stores, Frontend displays)
# ============================================================

class Event(BaseModel):
    """Structured event derived from tracking detections over time."""
    event_id: str = Field(..., description="Unique event ID")
    event_type: EventType = Field(..., description="Type of event")
    class_name: DetectionClass = Field(..., description="Object class involved")
    track_id: Optional[int] = Field(None, description="Tracking ID of involved object")
    start_time_sec: float = Field(..., description="Event start timestamp")
    end_time_sec: Optional[float] = Field(None, description="Event end timestamp")
    description: str = Field(..., description="Human-readable event description for NL search")
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[dict] = Field(default_factory=dict, description="Extra event data")


# ============================================================
# HEATMAP (ML generates, Backend serves, Frontend renders)
# ============================================================

class HeatmapData(BaseModel):
    """2D density matrix representing movement/detection frequency."""
    width: int = Field(..., description="Grid width (columns)")
    height: int = Field(..., description="Grid height (rows)")
    grid: List[List[float]] = Field(..., description="2D density values, normalized 0-1")
    video_width: int = Field(..., description="Original video pixel width")
    video_height: int = Field(..., description="Original video pixel height")


# ============================================================
# ML PIPELINE OUTPUT (ML -> Backend)
# ============================================================

class PipelineResult(BaseModel):
    """Complete output of ML pipeline for one video."""
    video_id: str
    total_frames: int
    fps_processed: float
    processing_time_sec: float
    detections: List[Detection]
    events: List[Event]
    heatmap: HeatmapData
    unique_person_count: int
    unique_vehicle_count: int
    object_class_counts: dict = Field(default_factory=dict, description="{'person': 12, 'car': 3, ...}")


# ============================================================
# API REQUEST/RESPONSE SCHEMAS (Backend <-> Frontend)
# ============================================================

class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    status: VideoStatus
    message: str


class VideoStatusResponse(BaseModel):
    video_id: str
    status: VideoStatus
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    current_step: Optional[str] = None
    estimated_remaining_sec: Optional[float] = None


class VideoSummary(BaseModel):
    video_id: str
    filename: str
    status: VideoStatus
    duration_sec: Optional[float] = None
    upload_time: datetime
    total_detections: int = 0
    unique_persons: int = 0
    unique_vehicles: int = 0
    thumbnail_path: Optional[str] = None


class NLQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query, e.g. 'Show me when a red truck entered'")
    video_id: Optional[str] = Field(None, description="Limit search to specific video")
    limit: int = Field(10, ge=1, le=50)


class NLQueryResult(BaseModel):
    event: Event
    video_id: str
    video_filename: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    thumbnail_path: Optional[str] = None


class NLQueryResponse(BaseModel):
    query: str
    results: List[NLQueryResult]
    total_results: int
    processing_time_ms: float


class AnalyticsSummary(BaseModel):
    total_videos: int
    total_processed: int
    total_processing: int
    total_detections: int
    total_unique_persons: int
    total_unique_vehicles: int
    total_events: int
    total_processing_hours: float


# ============================================================
# WEBSOCKET MESSAGES (Backend -> Frontend, real-time)
# ============================================================

class WSProgressMessage(BaseModel):
    """WebSocket message for real-time processing updates."""
    type: str = "progress"
    video_id: str
    status: VideoStatus
    progress_percent: float
    current_step: str
    message: Optional[str] = None
