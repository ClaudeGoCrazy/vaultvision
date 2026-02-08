"""
Auto Video Clip Extraction
Automatically extracts short video clips around detected events.
Produces shareable MP4 clips with event context.

Features:
- Extract clips with configurable padding before/after event
- Annotated clips with bounding boxes drawn
- Highlight reel generation (top N most interesting events)
- GIF generation for quick previews
"""
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ml.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


def extract_event_clip(
    video_path: str,
    video_id: str,
    event: dict,
    output_dir: Optional[str] = None,
    padding_before_sec: float = 2.0,
    padding_after_sec: float = 3.0,
    max_duration_sec: float = 15.0,
    draw_annotations: bool = False,
    detections: list[dict] | None = None,
) -> str | None:
    """
    Extract a short video clip around an event.

    Args:
        video_path: Source video path
        video_id: Video identifier
        event: Event dict with start_time_sec, end_time_sec
        output_dir: Where to save clips (default: OUTPUTS_DIR/video_id/clips)
        padding_before_sec: Seconds of context before event start
        padding_after_sec: Seconds of context after event end
        max_duration_sec: Maximum clip duration
        draw_annotations: Whether to draw bboxes on the clip
        detections: Detection dicts needed for annotations

    Returns:
        Path to saved clip, or None on failure
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps

    # Calculate clip boundaries (handle None end times)
    event_start = event.get("start_time_sec") or 0
    event_end = event.get("end_time_sec") or event_start
    start_sec = max(0, event_start - padding_before_sec)
    end_sec = min(video_duration, event_end + padding_after_sec)

    # Cap duration
    if end_sec - start_sec > max_duration_sec:
        end_sec = start_sec + max_duration_sec

    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)

    # Output path
    if output_dir is None:
        output_dir = str(OUTPUTS_DIR / video_id / "clips")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    event_id = event.get("event_id", "unknown")
    clip_filename = f"{event_id}.mp4"
    clip_path = str(Path(output_dir) / clip_filename)

    # Get frame dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        cap.release()
        return None

    h, w = test_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, video_fps, (w, h))

    # Build detection lookup for annotations
    det_by_frame = {}
    if draw_annotations and detections:
        for det in detections:
            fn = det["frame_number"]
            det_by_frame.setdefault(fn, []).append(det)

    # Write frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    frames_written = 0

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw annotations if requested
        if draw_annotations and frame_idx in det_by_frame:
            for det in det_by_frame[frame_idx]:
                bbox = det["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                label = f"{det['class_name']} {det.get('confidence', 0):.0%}"

                # Color based on event involvement
                color = (0, 255, 0)  # Green default
                if det.get("track_id") == event.get("track_id"):
                    color = (0, 0, 255)  # Red for event subject

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add timestamp overlay
        ts = frame_idx / video_fps
        ts_text = f"{ts:.1f}s"
        cv2.putText(frame, ts_text, (10, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame)
        frame_idx += 1
        frames_written += 1

    writer.release()
    cap.release()

    if frames_written == 0:
        Path(clip_path).unlink(missing_ok=True)
        return None

    logger.info(
        f"Extracted clip: {clip_filename} "
        f"({start_sec:.1f}s - {end_sec:.1f}s, {frames_written} frames)"
    )
    return clip_path


def extract_all_event_clips(
    video_path: str,
    video_id: str,
    events: list[dict],
    tracked_detections: list[dict] | None = None,
    max_clips: int = 20,
    draw_annotations: bool = True,
    padding_before_sec: float = 2.0,
    padding_after_sec: float = 3.0,
) -> dict[str, str]:
    """
    Extract clips for all events (up to max_clips).

    Priority: safety events > anomalies > other events

    Returns:
        Dict of event_id -> clip_path
    """
    # Prioritize events
    priority_order = {
        "fire_detected": 0,
        "smoke_detected": 0,
        "zone_intrusion": 1,
        "object_left": 1,
        "loitering": 2,
        "crowd_threshold": 2,
        "line_crossing": 3,
        "scene_anomaly": 3,
        "entry": 4,
        "exit": 4,
    }

    sorted_events = sorted(
        events,
        key=lambda e: (
            priority_order.get(e.get("event_type", ""), 5),
            -e.get("confidence", 0),
        ),
    )

    clip_map = {}
    for event in sorted_events[:max_clips]:
        clip_path = extract_event_clip(
            video_path=video_path,
            video_id=video_id,
            event=event,
            draw_annotations=draw_annotations,
            detections=tracked_detections,
            padding_before_sec=padding_before_sec,
            padding_after_sec=padding_after_sec,
        )
        if clip_path:
            clip_map[event["event_id"]] = clip_path

    logger.info(f"Extracted {len(clip_map)}/{len(events)} event clips for video {video_id}")
    return clip_map


def generate_highlight_reel(
    video_path: str,
    video_id: str,
    events: list[dict],
    tracked_detections: list[dict] | None = None,
    max_events: int = 10,
    clip_duration_sec: float = 5.0,
) -> str | None:
    """
    Generate a single highlight reel video combining top events.

    Returns:
        Path to highlight reel MP4, or None on failure
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps

    ret, test_frame = cap.read()
    if not ret:
        cap.release()
        return None
    h, w = test_frame.shape[:2]

    output_dir = OUTPUTS_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    reel_path = str(output_dir / "highlight_reel.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(reel_path, fourcc, video_fps, (w, h))

    # Sort events by importance
    priority_order = {
        "fire_detected": 0, "smoke_detected": 0,
        "zone_intrusion": 1, "object_left": 1,
        "loitering": 2, "crowd_threshold": 2,
    }

    sorted_events = sorted(
        events,
        key=lambda e: (priority_order.get(e.get("event_type", ""), 5), -e.get("confidence", 0)),
    )[:max_events]

    frames_written = 0
    for event in sorted_events:
        evt_start = event.get("start_time_sec") or 0
        start_sec = max(0, evt_start - 1.0)
        end_sec = min(video_duration, start_sec + clip_duration_sec)

        start_frame = int(start_sec * video_fps)
        end_frame = int(end_sec * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Add title card (1 second)
        title_frame = np.zeros((h, w, 3), dtype=np.uint8)
        event_type = event.get("event_type", "event").replace("_", " ").upper()
        cv2.putText(title_frame, event_type, (w // 6, h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        desc = event.get("description", "")[:60]
        cv2.putText(title_frame, desc, (20, h // 2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for _ in range(int(video_fps)):  # 1 second of title
            writer.write(title_frame)
            frames_written += 1

        # Write event frames
        for fi in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Overlay event info
            ts = fi / video_fps
            cv2.putText(frame, f"{ts:.1f}s | {event_type}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            writer.write(frame)
            frames_written += 1

    writer.release()
    cap.release()

    if frames_written == 0:
        Path(reel_path).unlink(missing_ok=True)
        return None

    logger.info(f"Generated highlight reel: {frames_written} frames from {len(sorted_events)} events")
    return reel_path
