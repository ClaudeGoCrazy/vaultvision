"""
Real-Time Streaming Pipeline
Processes live RTSP/webcam/file streams with continuous detection,
tracking, and event generation.

Features:
- RTSP, HTTP, webcam, and file input support
- Continuous object detection and tracking
- Real-time event generation and alerting
- Frame-skip for performance tuning
- Callback system for WebSocket/API integration
- Graceful start/stop with thread safety
"""
import logging
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ml.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_DEVICE,
    YOLO_IMGSZ,
)

logger = logging.getLogger(__name__)

# Lazy-loaded model cache
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO
                _model = YOLO(YOLO_MODEL)
                logger.info(f"Stream pipeline: Loaded {YOLO_MODEL} on {YOLO_DEVICE}")
    return _model


@dataclass
class StreamConfig:
    """Configuration for a stream processing session."""
    source: str  # RTSP URL, webcam index ("0"), or file path
    stream_id: str = ""
    camera_id: str = "cam_default"
    process_fps: float = 2.0  # Target processing FPS
    display: bool = False  # Show live window (for debugging)
    max_duration_sec: float = 0  # 0 = unlimited
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD
    classes_of_interest: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.stream_id:
            self.stream_id = f"stream_{uuid.uuid4().hex[:8]}"


@dataclass
class StreamDetection:
    """A single detection from the stream."""
    frame_number: int
    timestamp_sec: float
    class_name: str
    confidence: float
    bbox: dict  # {x1, y1, x2, y2}
    track_id: Optional[int] = None


class StreamCallbacks:
    """Callback handlers for stream events."""

    def __init__(self):
        self.on_detection: Optional[Callable[[list[StreamDetection]], None]] = None
        self.on_event: Optional[Callable[[dict], None]] = None
        self.on_frame: Optional[Callable[[int, np.ndarray, list[StreamDetection]], None]] = None
        self.on_status: Optional[Callable[[str, dict], None]] = None
        self.on_error: Optional[Callable[[str, Exception], None]] = None


# Extended COCO class name mapping
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 14: "bird", 15: "cat", 16: "dog", 24: "backpack",
    26: "handbag", 28: "suitcase", 56: "chair", 62: "laptop",
    63: "mouse", 64: "remote", 67: "cell_phone",
}


class StreamProcessor:
    """
    Real-time video stream processor.
    Runs detection + tracking on live feeds with callbacks.
    """

    def __init__(self, config: StreamConfig, callbacks: Optional[StreamCallbacks] = None):
        self.config = config
        self.callbacks = callbacks or StreamCallbacks()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Stats
        self.stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "total_detections": 0,
            "total_events": 0,
            "start_time": None,
            "fps_actual": 0.0,
            "class_counts": Counter(),
            "active_tracks": set(),
        }

        # Event detection state
        self._track_history: dict[int, deque] = {}
        self._track_last_seen: dict[int, float] = {}
        self._lost_track_timeout = 5.0  # seconds

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        """Start processing the stream in a background thread."""
        if self._running:
            logger.warning("Stream already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._process_loop,
            name=f"stream-{self.config.stream_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Stream started: {self.config.source} (id: {self.config.stream_id})")

        if self.callbacks.on_status:
            self.callbacks.on_status("started", {"stream_id": self.config.stream_id})

    def stop(self):
        """Stop the stream processor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info(f"Stream stopped: {self.config.stream_id}")

        if self.callbacks.on_status:
            self.callbacks.on_status("stopped", self.get_stats())

    def get_stats(self) -> dict:
        """Get current processing stats."""
        with self._lock:
            return {
                "stream_id": self.config.stream_id,
                "source": self.config.source,
                "is_running": self._running,
                "frames_processed": self.stats["frames_processed"],
                "frames_skipped": self.stats["frames_skipped"],
                "total_detections": self.stats["total_detections"],
                "total_events": self.stats["total_events"],
                "fps_actual": round(self.stats["fps_actual"], 1),
                "class_counts": dict(self.stats["class_counts"]),
                "active_tracks": len(self.stats["active_tracks"]),
                "uptime_sec": round(
                    time.time() - self.stats["start_time"], 1
                ) if self.stats["start_time"] else 0,
            }

    def _process_loop(self):
        """Main processing loop — runs in background thread."""
        source = self.config.source

        # Handle webcam index
        if source.isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Cannot open stream: {self.config.source}")
            if self.callbacks.on_error:
                self.callbacks.on_error("open_failed", Exception(f"Cannot open: {self.config.source}"))
            self._running = False
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / self.config.process_fps))

        self.stats["start_time"] = time.time()
        model = _get_model()

        frame_count = 0
        process_count = 0
        last_fps_time = time.time()
        fps_frame_count = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    # For files, this means end. For streams, could be temporary.
                    if isinstance(source, int) or "rtsp" in str(self.config.source).lower():
                        time.sleep(0.1)
                        continue
                    break

                frame_count += 1

                # Frame skip for performance
                if frame_count % frame_interval != 0:
                    with self._lock:
                        self.stats["frames_skipped"] += 1
                    continue

                timestamp_sec = frame_count / video_fps

                # Max duration check
                if self.config.max_duration_sec > 0 and timestamp_sec > self.config.max_duration_sec:
                    break

                # Run detection + tracking
                results = model.track(
                    frame,
                    persist=True,
                    conf=self.config.confidence_threshold,
                    iou=YOLO_IOU_THRESHOLD,
                    device=YOLO_DEVICE,
                    imgsz=YOLO_IMGSZ,
                    verbose=False,
                    tracker="bytetrack.yaml",
                )

                # Parse detections
                detections = []
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i])
                        class_name = COCO_NAMES.get(cls_id, f"class_{cls_id}")

                        if self.config.classes_of_interest and class_name not in self.config.classes_of_interest:
                            continue

                        conf = float(boxes.conf[i])
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        track_id = int(boxes.id[i]) if boxes.id is not None else None

                        det = StreamDetection(
                            frame_number=frame_count,
                            timestamp_sec=round(timestamp_sec, 3),
                            class_name=class_name,
                            confidence=round(conf, 3),
                            bbox={"x1": round(x1, 1), "y1": round(y1, 1),
                                  "x2": round(x2, 1), "y2": round(y2, 1)},
                            track_id=track_id,
                        )
                        detections.append(det)

                        # Track history for events
                        if track_id is not None:
                            if track_id not in self._track_history:
                                self._track_history[track_id] = deque(maxlen=100)
                                # New track event
                                self._emit_event({
                                    "event_type": "track_start",
                                    "class_name": class_name,
                                    "track_id": track_id,
                                    "timestamp_sec": timestamp_sec,
                                    "description": f"New {class_name} detected (track {track_id})",
                                })

                            self._track_history[track_id].append({
                                "timestamp": timestamp_sec,
                                "bbox": det.bbox,
                            })
                            self._track_last_seen[track_id] = timestamp_sec

                # Check for lost tracks
                current_track_ids = {d.track_id for d in detections if d.track_id is not None}
                for tid, last_seen in list(self._track_last_seen.items()):
                    if tid not in current_track_ids and timestamp_sec - last_seen > self._lost_track_timeout:
                        self._emit_event({
                            "event_type": "track_end",
                            "track_id": tid,
                            "timestamp_sec": timestamp_sec,
                            "description": f"Track {tid} lost (last seen {last_seen:.1f}s)",
                        })
                        del self._track_last_seen[tid]

                # Update stats
                with self._lock:
                    self.stats["frames_processed"] += 1
                    self.stats["total_detections"] += len(detections)
                    self.stats["active_tracks"] = current_track_ids
                    for d in detections:
                        self.stats["class_counts"][d.class_name] += 1

                # Calculate actual FPS
                fps_frame_count += 1
                now = time.time()
                if now - last_fps_time >= 2.0:
                    self.stats["fps_actual"] = fps_frame_count / (now - last_fps_time)
                    fps_frame_count = 0
                    last_fps_time = now

                # Fire callbacks
                if detections and self.callbacks.on_detection:
                    self.callbacks.on_detection(detections)

                if self.callbacks.on_frame:
                    self.callbacks.on_frame(frame_count, frame, detections)

                # Display window (debug mode)
                if self.config.display:
                    display_frame = frame.copy()
                    for d in detections:
                        x1, y1 = int(d.bbox["x1"]), int(d.bbox["y1"])
                        x2, y2 = int(d.bbox["x2"]), int(d.bbox["y2"])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{d.class_name} {d.confidence:.0%}"
                        if d.track_id:
                            label += f" #{d.track_id}"
                        cv2.putText(display_frame, label, (x1, y1 - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.imshow(f"VaultVision: {self.config.stream_id}", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                process_count += 1

        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            if self.callbacks.on_error:
                self.callbacks.on_error("processing_error", e)
        finally:
            cap.release()
            if self.config.display:
                cv2.destroyAllWindows()
            self._running = False
            logger.info(
                f"Stream {self.config.stream_id} ended: "
                f"{process_count} frames processed, "
                f"{self.stats['total_detections']} detections"
            )

    def _emit_event(self, event: dict):
        """Emit an event through the callback system."""
        event.setdefault("stream_id", self.config.stream_id)
        event.setdefault("camera_id", self.config.camera_id)

        with self._lock:
            self.stats["total_events"] += 1

        if self.callbacks.on_event:
            try:
                self.callbacks.on_event(event)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")


class StreamManager:
    """
    Manages multiple concurrent stream processors.
    """

    def __init__(self):
        self._streams: dict[str, StreamProcessor] = {}
        self._lock = threading.Lock()

    def add_stream(
        self,
        source: str,
        camera_id: str = "",
        callbacks: Optional[StreamCallbacks] = None,
        **config_kwargs,
    ) -> str:
        """
        Add and start a new stream.

        Returns:
            Stream ID
        """
        config = StreamConfig(
            source=source,
            camera_id=camera_id or f"cam_{len(self._streams)}",
            **config_kwargs,
        )

        processor = StreamProcessor(config, callbacks)

        with self._lock:
            self._streams[config.stream_id] = processor

        processor.start()
        return config.stream_id

    def stop_stream(self, stream_id: str):
        """Stop a specific stream."""
        with self._lock:
            proc = self._streams.get(stream_id)
        if proc:
            proc.stop()

    def stop_all(self):
        """Stop all streams."""
        with self._lock:
            stream_ids = list(self._streams.keys())
        for sid in stream_ids:
            self.stop_stream(sid)

    def get_stream_stats(self, stream_id: str) -> dict | None:
        """Get stats for a specific stream."""
        proc = self._streams.get(stream_id)
        return proc.get_stats() if proc else None

    def get_all_stats(self) -> dict:
        """Get stats for all streams."""
        return {
            sid: proc.get_stats()
            for sid, proc in self._streams.items()
        }

    def list_streams(self) -> list[dict]:
        """List all streams with basic info."""
        return [
            {
                "stream_id": sid,
                "source": proc.config.source,
                "camera_id": proc.config.camera_id,
                "is_running": proc.is_running,
            }
            for sid, proc in self._streams.items()
        ]


# Singleton manager
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
