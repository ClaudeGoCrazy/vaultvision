"""
YOLOv8 Object Detection Module
Runs inference on extracted frames and returns Detection objects.
"""
import logging
from pathlib import Path
from ultralytics import YOLO

from ml.config import (
    YOLO_MODEL,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IOU_THRESHOLD,
    YOLO_DEVICE,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

# Map YOLO COCO class names to our DetectionClass enum values
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
}


class ObjectDetector:
    """YOLOv8-based object detector."""

    def __init__(self, model_name: str = YOLO_MODEL, device: str = YOLO_DEVICE):
        logger.info(f"Loading YOLOv8 model: {model_name} on device: {device}")
        self.model = YOLO(model_name)
        self.device = device
        self.model_name = model_name
        logger.info("YOLOv8 model loaded successfully")

    def detect_batch(
        self,
        frame_paths: list[Path],
        frame_timestamps: list[float],
        confidence: float = YOLO_CONFIDENCE_THRESHOLD,
        iou: float = YOLO_IOU_THRESHOLD,
        progress_callback=None,
    ) -> list[dict]:
        """
        Run detection on a batch of frames.

        Returns:
            List of raw detection dicts (before tracking):
            [
                {
                    "frame_number": int,
                    "timestamp_sec": float,
                    "class_name": str,
                    "confidence": float,
                    "bbox": {"x1": float, "y1": float, "x2": float, "y2": float},
                },
                ...
            ]
        """
        all_detections = []
        total = len(frame_paths)

        # Process in batches for GPU efficiency
        batch_size = 16
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_paths = [str(p) for p in frame_paths[batch_start:batch_end]]

            results = self.model(
                batch_paths,
                conf=confidence,
                iou=iou,
                device=self.device,
                verbose=False,
            )

            for i, result in enumerate(results):
                frame_idx = batch_start + i
                timestamp = frame_timestamps[frame_idx]

                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Map to our detection class
                    mapped_class = COCO_TO_DETECTION_CLASS.get(cls_name, "other")

                    all_detections.append({
                        "frame_number": frame_idx,
                        "timestamp_sec": timestamp,
                        "class_name": mapped_class,
                        "confidence": conf,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })

            if progress_callback:
                pct = (batch_end / total) * 100
                progress_callback(pct, f"Detection: {batch_end}/{total} frames processed")

        logger.info(f"Detected {len(all_detections)} objects across {total} frames")
        return all_detections
