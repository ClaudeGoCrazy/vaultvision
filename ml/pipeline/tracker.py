"""
Multi-Object Tracker using ByteTrack via Ultralytics.
Assigns persistent track IDs to detections across frames.

Performance optimizations:
- Singleton model cache (avoids reloading on each call)
- Batch-aware processing with progress reporting
"""
import logging
import uuid
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from ml.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_DEVICE,
    YOLO_IMGSZ,
    YOLO_MAX_DETECTIONS,
)

logger = logging.getLogger(__name__)

# Extended COCO class mapping — covers all 80 COCO classes that map to our schema
COCO_TO_DETECTION_CLASS = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "dog": "dog",
    "cat": "cat",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "cell phone": "cell_phone",
    # Additional vehicle-like mappings
    "train": "vehicle",
    "boat": "vehicle",
    "airplane": "vehicle",
}

# Singleton model cache to avoid reloading on every pipeline run
_model_cache: dict[str, YOLO] = {}


def _get_model(model_name: str = YOLO_MODEL) -> YOLO:
    """Get or create a cached YOLO model instance."""
    if model_name not in _model_cache:
        logger.info(f"Loading YOLOv8 model: {model_name} (will be cached for reuse)")
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


def clear_model_cache():
    """Release cached models to free memory."""
    _model_cache.clear()
    logger.info("Model cache cleared")


class MultiObjectTracker:
    """
    Tracks objects across video frames using Ultralytics built-in ByteTrack.
    Produces tracked detections with persistent IDs and track summaries.
    """

    def __init__(self, model_name: str = YOLO_MODEL, device: str = YOLO_DEVICE):
        self.model = _get_model(model_name)
        self.device = device

    def track_video(
        self,
        video_path: str,
        frame_paths: list[Path],
        frame_timestamps: list[float],
        confidence: float = YOLO_CONFIDENCE_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
        progress_callback=None,
    ) -> tuple[list[dict], dict]:
        """
        Run detection + tracking on extracted frames.

        Returns:
            (tracked_detections, track_summaries)
            - tracked_detections: list of detection dicts with track_id
            - track_summaries: dict of track_id -> {class, times, bbox history}
        """
        all_detections = []
        track_summaries = defaultdict(lambda: {
            "class_name": None,
            "first_seen_sec": float("inf"),
            "last_seen_sec": 0.0,
            "first_frame": float("inf"),
            "last_frame": 0,
            "frame_count": 0,
            "bbox_history": [],  # For spatial analysis
        })

        total_frames = len(frame_paths)

        for idx, frame_path in enumerate(frame_paths):
            timestamp = frame_timestamps[idx]

            results = self.model.track(
                str(frame_path),
                conf=confidence,
                iou=iou,
                device=self.device,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                imgsz=YOLO_IMGSZ,
                max_det=YOLO_MAX_DETECTIONS,
            )

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    track_id = int(box.id[0]) if box.id is not None else None
                    mapped_class = COCO_TO_DETECTION_CLASS.get(cls_name, "other")

                    bbox_dict = {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    }

                    detection = {
                        "detection_id": str(uuid.uuid4()),
                        "frame_number": idx,
                        "timestamp_sec": round(timestamp, 3),
                        "class_name": mapped_class,
                        "confidence": round(conf, 4),
                        "bbox": bbox_dict,
                        "track_id": track_id,
                    }
                    all_detections.append(detection)

                    # Update track summary with spatial data
                    if track_id is not None:
                        summary = track_summaries[track_id]
                        summary["class_name"] = mapped_class
                        summary["first_seen_sec"] = min(summary["first_seen_sec"], timestamp)
                        summary["last_seen_sec"] = max(summary["last_seen_sec"], timestamp)
                        summary["first_frame"] = min(summary["first_frame"], idx)
                        summary["last_frame"] = max(summary["last_frame"], idx)
                        summary["frame_count"] += 1
                        summary["bbox_history"].append({
                            "frame": idx,
                            "timestamp": timestamp,
                            "bbox": bbox_dict,
                        })

            if progress_callback and (idx % 5 == 0 or idx == total_frames - 1):
                pct = ((idx + 1) / total_frames) * 100
                progress_callback(pct, f"Tracking: frame {idx + 1}/{total_frames}")

        logger.info(
            f"Tracking complete: {len(all_detections)} detections, "
            f"{len(track_summaries)} unique tracks"
        )
        return all_detections, dict(track_summaries)
