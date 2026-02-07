"""
Thumbnail / Keyframe Extraction
Extracts JPEG thumbnails from video at specific timestamps for events and tracks.
"""
import cv2
import logging
from pathlib import Path

from ml.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def extract_thumbnail(
    video_path: str,
    timestamp_sec: float,
    output_path: str,
    width: int = 320,
    bbox: dict | None = None,
) -> str | None:
    """
    Extract a single thumbnail frame from video at given timestamp.

    Args:
        video_path: Path to video file
        timestamp_sec: Time in seconds to extract frame
        output_path: Where to save the JPEG
        width: Target thumbnail width (height auto-calculated)
        bbox: Optional bounding box to crop around {x1, y1, x2, y2}

    Returns:
        Path to saved thumbnail, or None on failure
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Crop to bounding box region with padding if specified
    if bbox:
        h, w = frame.shape[:2]
        pad = 30  # pixels of context around the detection
        x1 = max(0, int(bbox["x1"]) - pad)
        y1 = max(0, int(bbox["y1"]) - pad)
        x2 = min(w, int(bbox["x2"]) + pad)
        y2 = min(h, int(bbox["y2"]) + pad)
        frame = frame[y1:y2, x1:x2]

    # Resize to target width, maintaining aspect ratio
    h, w = frame.shape[:2]
    if w > 0:
        scale = width / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (width, new_h))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return output_path


def extract_event_thumbnails(
    video_path: str,
    video_id: str,
    events: list[dict],
    tracked_detections: list[dict] | None = None,
    max_thumbnails: int = 50,
) -> dict[str, str]:
    """
    Extract thumbnails for each event at its start timestamp.

    Args:
        video_path: Path to video file
        video_id: Video ID for organizing output
        events: List of event dicts
        tracked_detections: Optional — used to find bbox for cropped thumbnails
        max_thumbnails: Cap to avoid disk bloat on long videos

    Returns:
        Dict of event_id -> thumbnail_path
    """
    output_dir = OUTPUTS_DIR / video_id / "thumbnails"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a lookup: (frame_number, track_id) -> bbox for cropped thumbnails
    bbox_lookup = {}
    if tracked_detections:
        for det in tracked_detections:
            key = (det["frame_number"], det.get("track_id"))
            bbox_lookup[key] = det["bbox"]

    thumbnail_map = {}
    count = 0

    for event in events:
        if count >= max_thumbnails:
            break

        event_id = event["event_id"]
        ts = event["start_time_sec"]

        # Full-frame thumbnail
        full_path = str(output_dir / f"{event_id}_full.jpg")
        result = extract_thumbnail(video_path, ts, full_path, width=640)
        if result:
            thumbnail_map[event_id] = result
            count += 1

        # Cropped thumbnail around the detected object (if we have bbox data)
        track_id = event.get("track_id")
        if track_id is not None and tracked_detections:
            # Find the closest detection to this event's timestamp
            closest_det = None
            min_dt = float("inf")
            for det in tracked_detections:
                if det.get("track_id") == track_id:
                    dt = abs(det["timestamp_sec"] - ts)
                    if dt < min_dt:
                        min_dt = dt
                        closest_det = det

            if closest_det and count < max_thumbnails:
                crop_path = str(output_dir / f"{event_id}_crop.jpg")
                result = extract_thumbnail(
                    video_path, ts, crop_path, width=200, bbox=closest_det["bbox"]
                )
                if result:
                    thumbnail_map[f"{event_id}_crop"] = result
                    count += 1

    logger.info(f"Extracted {count} thumbnails for {len(events)} events (video {video_id})")
    return thumbnail_map


def extract_video_thumbnail(video_path: str, video_id: str) -> str | None:
    """Extract a single representative thumbnail from the video (frame at 10% duration)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(1, int(total_frames * 0.1))  # 10% into the video
    cap.release()

    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    ts = target_frame / fps if fps > 0 else 1.0

    output_dir = OUTPUTS_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = str(output_dir / "thumbnail.jpg")

    return extract_thumbnail(video_path, ts, thumb_path, width=480)
