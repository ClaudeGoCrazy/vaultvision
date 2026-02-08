"""
Object Attribute Extraction
Extracts visual attributes from detection crops:
- Dominant clothing colors (people)
- Vehicle color and type classification
- Size classification (small/medium/large)
- Brightness/visibility conditions
"""
import logging
from collections import Counter
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Named color ranges in HSV space
COLOR_RANGES = {
    "red": [(0, 70, 50, 10, 255, 255), (170, 70, 50, 180, 255, 255)],
    "orange": [(10, 70, 50, 25, 255, 255)],
    "yellow": [(25, 70, 50, 35, 255, 255)],
    "green": [(35, 70, 50, 85, 255, 255)],
    "cyan": [(85, 70, 50, 100, 255, 255)],
    "blue": [(100, 70, 50, 130, 255, 255)],
    "purple": [(130, 70, 50, 170, 255, 255)],
    "white": [(0, 0, 180, 180, 30, 255)],
    "gray": [(0, 0, 80, 180, 30, 180)],
    "black": [(0, 0, 0, 180, 30, 80)],
    "brown": [(10, 50, 30, 25, 200, 150)],
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}


def _classify_color(crop: np.ndarray) -> tuple[str, list[tuple[str, float]]]:
    """
    Classify the dominant color of a crop using HSV analysis.

    Returns:
        (dominant_color_name, [(color_name, percentage), ...])
    """
    if crop.size == 0:
        return "unknown", []

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    if total_pixels == 0:
        return "unknown", []

    color_percentages = []

    for color_name, ranges in COLOR_RANGES.items():
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in ranges:
            if len(r) == 6:
                lower = np.array(r[:3])
                upper = np.array(r[3:])
                mask = cv2.inRange(hsv, lower, upper)
                total_mask = cv2.bitwise_or(total_mask, mask)

        count = cv2.countNonZero(total_mask)
        pct = count / total_pixels
        if pct > 0.05:  # At least 5% of pixels
            color_percentages.append((color_name, round(pct, 3)))

    color_percentages.sort(key=lambda x: x[1], reverse=True)

    dominant = color_percentages[0][0] if color_percentages else "unknown"
    return dominant, color_percentages


def _classify_size(bbox: dict, frame_width: int, frame_height: int) -> str:
    """Classify object size relative to frame."""
    obj_w = bbox["x2"] - bbox["x1"]
    obj_h = bbox["y2"] - bbox["y1"]
    obj_area = obj_w * obj_h
    frame_area = frame_width * frame_height

    if frame_area == 0:
        return "unknown"

    ratio = obj_area / frame_area

    if ratio > 0.15:
        return "large"
    elif ratio > 0.03:
        return "medium"
    else:
        return "small"


def _classify_brightness(crop: np.ndarray) -> str:
    """Classify the brightness/visibility of a crop."""
    if crop.size == 0:
        return "unknown"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    if mean_brightness > 180:
        return "bright"
    elif mean_brightness > 80:
        return "normal"
    else:
        return "dark"


def _extract_person_attributes(crop: np.ndarray) -> dict:
    """Extract attributes specific to a person detection."""
    h = crop.shape[0]

    # Split into upper body (clothing) and lower body
    upper = crop[:h // 2]
    lower = crop[h // 2:]

    upper_color, upper_colors = _classify_color(upper)
    lower_color, lower_colors = _classify_color(lower)

    return {
        "upper_body_color": upper_color,
        "upper_body_colors": upper_colors[:3],
        "lower_body_color": lower_color,
        "lower_body_colors": lower_colors[:3],
    }


def _extract_vehicle_attributes(crop: np.ndarray, class_name: str) -> dict:
    """Extract attributes specific to a vehicle detection."""
    color, colors = _classify_color(crop)

    # Simple vehicle type mapping
    vehicle_type = "unknown"
    if class_name == "car":
        vehicle_type = "sedan"  # default, could be enhanced with model
    elif class_name == "truck":
        vehicle_type = "truck"
    elif class_name == "bus":
        vehicle_type = "bus"
    elif class_name == "motorcycle":
        vehicle_type = "motorcycle"
    elif class_name == "bicycle":
        vehicle_type = "bicycle"

    return {
        "vehicle_color": color,
        "vehicle_colors": colors[:3],
        "vehicle_type": vehicle_type,
    }


def extract_attributes(
    video_path: str,
    tracked_detections: list[dict],
    track_summaries: dict,
    video_width: int,
    video_height: int,
    sample_rate: int = 3,
) -> dict:
    """
    Extract visual attributes for all tracked objects.

    Args:
        video_path: Path to video file
        tracked_detections: Detections with track_id, bbox, class_name
        track_summaries: Track metadata
        video_width: Frame width
        video_height: Frame height
        sample_rate: Process every N detections per track

    Returns:
        {
            "track_attributes": {
                track_id: {
                    "class_name": str,
                    "dominant_color": str,
                    "size": str,
                    "brightness": str,
                    "colors": [(name, pct), ...],
                    "person_attrs": {...} or None,
                    "vehicle_attrs": {...} or None,
                }
            },
            "color_summary": {"red": n, "blue": n, ...},
            "size_summary": {"small": n, "medium": n, "large": n},
        }
    """
    # Group detections by track
    tracks: dict[int, list[dict]] = {}
    for det in tracked_detections:
        tid = det.get("track_id")
        if tid is not None:
            tracks.setdefault(tid, []).append(det)

    # Determine which frames we need
    frames_needed: dict[int, list[tuple[int, dict]]] = {}
    for tid, dets in tracks.items():
        # Sample the middle detection (most representative)
        sampled = dets[len(dets) // 2: len(dets) // 2 + 1]
        if not sampled:
            sampled = dets[:1]
        for det in sampled:
            fn = det["frame_number"]
            frames_needed.setdefault(fn, []).append((tid, det))

    if not frames_needed:
        return {"track_attributes": {}, "color_summary": {}, "size_summary": {}}

    # Read frames and extract attributes
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video for attributes: {video_path}")
        return {"track_attributes": {}, "color_summary": {}, "size_summary": {}}

    track_attrs: dict[int, dict] = {}
    color_counter = Counter()
    size_counter = Counter()

    for target_frame in sorted(frames_needed.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        fh, fw = frame.shape[:2]

        for tid, det in frames_needed[target_frame]:
            bbox = det["bbox"]
            x1 = max(0, int(bbox["x1"]))
            y1 = max(0, int(bbox["y1"]))
            x2 = min(fw, int(bbox["x2"]))
            y2 = min(fh, int(bbox["y2"]))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            class_name = det["class_name"]

            # General attributes
            dominant_color, color_list = _classify_color(crop)
            size = _classify_size(bbox, video_width, video_height)
            brightness = _classify_brightness(crop)

            attrs = {
                "class_name": class_name,
                "dominant_color": dominant_color,
                "size": size,
                "brightness": brightness,
                "colors": color_list[:5],
                "person_attrs": None,
                "vehicle_attrs": None,
            }

            # Class-specific attributes
            if class_name == "person":
                attrs["person_attrs"] = _extract_person_attributes(crop)
            elif class_name in VEHICLE_CLASSES:
                attrs["vehicle_attrs"] = _extract_vehicle_attributes(crop, class_name)

            track_attrs[tid] = attrs
            color_counter[dominant_color] += 1
            size_counter[size] += 1

    cap.release()

    result = {
        "track_attributes": track_attrs,
        "color_summary": dict(color_counter),
        "size_summary": dict(size_counter),
    }

    logger.info(
        f"Attributes: Extracted for {len(track_attrs)} tracks, "
        f"top colors: {color_counter.most_common(3)}"
    )

    return result
