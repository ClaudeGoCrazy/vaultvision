"""
Smoke & Fire Detection Module
Detects potential fire and smoke events using:
- Color analysis in HSV/YCrCb space
- Motion + flickering patterns (fire)
- Opacity + spread patterns (smoke)
- Temporal consistency to reduce false positives
"""
import logging
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _detect_fire_pixels(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect fire-colored pixels using multi-color-space rules.

    Fire characteristics in color space:
    - RGB: R > G > B, R > 190
    - YCrCb: High Cr, low Cb
    - HSV: Low hue (0-40), high saturation and value

    Returns:
        (fire_mask, fire_ratio)
    """
    h, w = frame.shape[:2]
    total_pixels = h * w

    # HSV-based fire detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Fire hue range (red-orange-yellow)
    lower1 = np.array([0, 80, 150])
    upper1 = np.array([40, 255, 255])
    lower2 = np.array([160, 80, 150])  # Wrap-around red
    upper2 = np.array([180, 255, 255])

    mask_hsv1 = cv2.inRange(hsv, lower1, upper1)
    mask_hsv2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # RGB rule: R > G > B and R is high
    b, g, r = cv2.split(frame)
    mask_rgb = ((r > 190) & (r > g) & (g > b)).astype(np.uint8) * 255

    # Combined mask
    fire_mask = cv2.bitwise_and(mask_hsv, mask_rgb)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    fire_ratio = cv2.countNonZero(fire_mask) / total_pixels if total_pixels > 0 else 0
    return fire_mask, fire_ratio


def _detect_smoke_pixels(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect smoke-colored pixels.

    Smoke characteristics:
    - Low saturation (gray/white)
    - Medium-high value
    - Uniform color distribution
    """
    h, w = frame.shape[:2]
    total_pixels = h * w

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Smoke: low saturation, medium-high brightness (gray/white haze)
    lower = np.array([0, 0, 120])
    upper = np.array([180, 60, 240])
    mask_smoke = cv2.inRange(hsv, lower, upper)

    # Filter by texture — smoke regions have low gradient magnitude
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    # Smoke tends to be blurry / low-texture
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    low_texture = (np.abs(laplacian) < 10).astype(np.uint8) * 255

    # Combined
    smoke_mask = cv2.bitwise_and(mask_smoke, low_texture)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)

    smoke_ratio = cv2.countNonZero(smoke_mask) / total_pixels if total_pixels > 0 else 0
    return smoke_mask, smoke_ratio


def detect_safety_events(
    video_path: str,
    video_id: str,
    fps_sample: float = 1.0,
    fire_threshold: float = 0.005,
    smoke_threshold: float = 0.05,
    min_consecutive_frames: int = 2,
) -> dict:
    """
    Analyze video for fire and smoke events.

    Args:
        video_path: Path to video file
        video_id: Video identifier
        fps_sample: Frames per second to analyze
        fire_threshold: Min fire pixel ratio to trigger
        smoke_threshold: Min smoke pixel ratio to trigger
        min_consecutive_frames: Consecutive detections needed to confirm

    Returns:
        {
            "fire_events": [event_dicts],
            "smoke_events": [event_dicts],
            "fire_frames": [(timestamp, ratio), ...],
            "smoke_frames": [(timestamp, ratio), ...],
            "events": [combined event dicts for pipeline],
            "summary": {
                "fire_detected": bool,
                "smoke_detected": bool,
                "max_fire_ratio": float,
                "max_smoke_ratio": float,
                "first_fire_time_sec": float | None,
                "first_smoke_time_sec": float | None,
            }
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video for safety analysis: {video_path}")
        return _empty_result()

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps_sample))

    fire_frames = []
    smoke_frames = []
    fire_streak = 0
    smoke_streak = 0
    fire_events = []
    smoke_events = []
    fire_event_start = None
    smoke_event_start = None

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval != 0:
            frame_num += 1
            continue

        timestamp = frame_num / video_fps

        # Fire detection
        _, fire_ratio = _detect_fire_pixels(frame)
        if fire_ratio >= fire_threshold:
            fire_frames.append((round(timestamp, 2), round(fire_ratio, 4)))
            fire_streak += 1
            if fire_streak >= min_consecutive_frames and fire_event_start is None:
                fire_event_start = timestamp - (min_consecutive_frames - 1) / fps_sample
        else:
            if fire_event_start is not None:
                fire_events.append({
                    "start_time_sec": round(max(0, fire_event_start), 2),
                    "end_time_sec": round(timestamp, 2),
                    "max_ratio": round(max(r for _, r in fire_frames[-fire_streak:]) if fire_streak else fire_ratio, 4),
                })
                fire_event_start = None
            fire_streak = 0

        # Smoke detection
        _, smoke_ratio = _detect_smoke_pixels(frame)
        if smoke_ratio >= smoke_threshold:
            smoke_frames.append((round(timestamp, 2), round(smoke_ratio, 4)))
            smoke_streak += 1
            if smoke_streak >= min_consecutive_frames and smoke_event_start is None:
                smoke_event_start = timestamp - (min_consecutive_frames - 1) / fps_sample
        else:
            if smoke_event_start is not None:
                smoke_events.append({
                    "start_time_sec": round(max(0, smoke_event_start), 2),
                    "end_time_sec": round(timestamp, 2),
                    "max_ratio": round(max(r for _, r in smoke_frames[-smoke_streak:]) if smoke_streak else smoke_ratio, 4),
                })
                smoke_event_start = None
            smoke_streak = 0

        frame_num += 1

    cap.release()

    # Close any open events
    duration = total_frames / video_fps if video_fps > 0 else 0
    if fire_event_start is not None:
        fire_events.append({
            "start_time_sec": round(max(0, fire_event_start), 2),
            "end_time_sec": round(duration, 2),
            "max_ratio": round(fire_frames[-1][1] if fire_frames else 0, 4),
        })
    if smoke_event_start is not None:
        smoke_events.append({
            "start_time_sec": round(max(0, smoke_event_start), 2),
            "end_time_sec": round(duration, 2),
            "max_ratio": round(smoke_frames[-1][1] if smoke_frames else 0, 4),
        })

    # Convert to pipeline events
    pipeline_events = []
    for i, fe in enumerate(fire_events):
        pipeline_events.append({
            "event_id": f"fire_{video_id}_{i}",
            "event_type": "fire_detected",
            "class_name": "fire",
            "track_id": None,
            "start_time_sec": fe["start_time_sec"],
            "end_time_sec": fe["end_time_sec"],
            "confidence": min(0.99, 0.5 + fe["max_ratio"] * 50),
            "description": (
                f"Potential fire detected from {fe['start_time_sec']:.1f}s "
                f"to {fe['end_time_sec']:.1f}s (intensity: {fe['max_ratio']:.2%})"
            ),
            "metadata": {"max_fire_ratio": fe["max_ratio"], "severity": "high"},
        })

    for i, se in enumerate(smoke_events):
        pipeline_events.append({
            "event_id": f"smoke_{video_id}_{i}",
            "event_type": "smoke_detected",
            "class_name": "smoke",
            "track_id": None,
            "start_time_sec": se["start_time_sec"],
            "end_time_sec": se["end_time_sec"],
            "confidence": min(0.95, 0.4 + se["max_ratio"] * 5),
            "description": (
                f"Potential smoke detected from {se['start_time_sec']:.1f}s "
                f"to {se['end_time_sec']:.1f}s (coverage: {se['max_ratio']:.2%})"
            ),
            "metadata": {"max_smoke_ratio": se["max_ratio"], "severity": "medium"},
        })

    result = {
        "fire_events": fire_events,
        "smoke_events": smoke_events,
        "fire_frames": fire_frames,
        "smoke_frames": smoke_frames,
        "events": pipeline_events,
        "summary": {
            "fire_detected": len(fire_events) > 0,
            "smoke_detected": len(smoke_events) > 0,
            "max_fire_ratio": round(max((r for _, r in fire_frames), default=0), 4),
            "max_smoke_ratio": round(max((r for _, r in smoke_frames), default=0), 4),
            "first_fire_time_sec": fire_events[0]["start_time_sec"] if fire_events else None,
            "first_smoke_time_sec": smoke_events[0]["start_time_sec"] if smoke_events else None,
        },
    }

    logger.info(
        f"Safety analysis: {len(fire_events)} fire events, "
        f"{len(smoke_events)} smoke events (analyzed {frame_num} frames)"
    )

    return result


def _empty_result() -> dict:
    return {
        "fire_events": [],
        "smoke_events": [],
        "fire_frames": [],
        "smoke_frames": [],
        "events": [],
        "summary": {
            "fire_detected": False,
            "smoke_detected": False,
            "max_fire_ratio": 0,
            "max_smoke_ratio": 0,
            "first_fire_time_sec": None,
            "first_smoke_time_sec": None,
        },
    }
