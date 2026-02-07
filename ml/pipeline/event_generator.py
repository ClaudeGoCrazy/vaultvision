"""
Event Generator
Analyzes tracked detections to produce structured Event objects.
Integrates basic entry/exit events with the full anomaly detection module.
"""
import uuid
import logging
from collections import defaultdict

from ml.config import LOITERING_THRESHOLD_SEC, CROWD_THRESHOLD
from ml.analytics.anomaly import (
    Zone,
    detect_zone_intrusions,
    detect_loitering_spatial,
    detect_object_left_behind,
    detect_crowd_density,
)

logger = logging.getLogger(__name__)


def _format_timestamp(seconds: float) -> str:
    """Format seconds into M:SS."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def _class_display_name(class_name: str) -> str:
    """Convert class_name enum value to display-friendly name."""
    return class_name.replace("_", " ").title()


def generate_events(
    tracked_detections: list[dict],
    track_summaries: dict,
    video_width: int = 1920,
    video_height: int = 1080,
    zones: list[Zone] | None = None,
) -> list[dict]:
    """
    Generate all structured events from tracked detections.
    Combines entry/exit, loitering, crowd, zone intrusion, and object-left events.

    Args:
        tracked_detections: List of detection dicts with track_id assigned
        track_summaries: Dict of track_id -> summary info
        video_width: Video width for spatial calculations
        video_height: Video height for spatial calculations
        zones: Optional list of Zone objects for zone-based detection

    Returns:
        List of Event dicts matching shared schema
    """
    events = []

    # --- Entry/Exit events from track summaries ---
    for track_id, summary in track_summaries.items():
        cls_name = summary["class_name"]
        display_name = _class_display_name(cls_name)
        first_ts = summary["first_seen_sec"]
        last_ts = summary["last_seen_sec"]

        # Entry event
        events.append({
            "event_id": str(uuid.uuid4()),
            "event_type": "entry",
            "class_name": cls_name,
            "track_id": track_id,
            "start_time_sec": round(first_ts, 3),
            "end_time_sec": None,
            "description": (
                f"{display_name} (Track #{track_id}) entered the frame "
                f"at {_format_timestamp(first_ts)}"
            ),
            "confidence": 0.9,
            "metadata": {"source": "tracking"},
        })

        # Exit event (if track spans multiple frames)
        if last_ts > first_ts:
            duration = last_ts - first_ts
            events.append({
                "event_id": str(uuid.uuid4()),
                "event_type": "exit",
                "class_name": cls_name,
                "track_id": track_id,
                "start_time_sec": round(last_ts, 3),
                "end_time_sec": None,
                "description": (
                    f"{display_name} (Track #{track_id}) exited the frame "
                    f"at {_format_timestamp(last_ts)} "
                    f"(visible for {duration:.1f}s)"
                ),
                "confidence": 0.85,
                "metadata": {"duration_sec": round(duration, 2)},
            })

    # --- Advanced anomaly detection ---

    # Spatially-aware loitering (replaces simple time-based check)
    loitering_events = detect_loitering_spatial(
        tracked_detections=tracked_detections,
        track_summaries=track_summaries,
        video_width=video_width,
        video_height=video_height,
    )
    events.extend(loitering_events)

    # Crowd density detection (improved with continuous periods)
    crowd_events = detect_crowd_density(
        tracked_detections=tracked_detections,
        zones=zones,
        threshold=CROWD_THRESHOLD,
    )
    events.extend(crowd_events)

    # Object left behind detection
    object_left_events = detect_object_left_behind(
        tracked_detections=tracked_detections,
        track_summaries=track_summaries,
    )
    events.extend(object_left_events)

    # Zone intrusion detection (if zones configured)
    if zones:
        zone_events = detect_zone_intrusions(
            tracked_detections=tracked_detections,
            track_summaries=track_summaries,
            zones=zones,
        )
        events.extend(zone_events)

    logger.info(
        f"Generated {len(events)} total events: "
        f"entry/exit={sum(1 for e in events if e['event_type'] in ('entry','exit'))}, "
        f"loitering={sum(1 for e in events if e['event_type'] == 'loitering')}, "
        f"crowd={sum(1 for e in events if e['event_type'] == 'crowd_threshold')}, "
        f"zone_intrusion={sum(1 for e in events if e['event_type'] == 'zone_intrusion')}, "
        f"object_left={sum(1 for e in events if e['event_type'] == 'object_left')}"
    )
    return events
