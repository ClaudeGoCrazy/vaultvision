"""
Anomaly Detection Module
Configurable zone intrusion, loitering with spatial analysis,
crowd density per zone, and object-left-behind detection.
"""
import uuid
import math
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from ml.config import (
    LOITERING_THRESHOLD_SEC,
    LOITERING_SPATIAL_THRESH,
    CROWD_THRESHOLD,
    OBJECT_LEFT_THRESHOLD_SEC,
    OBJECT_LEFT_CLASSES,
)

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A polygon zone defined by a list of (x, y) vertices in pixel coordinates."""
    zone_id: str
    name: str
    vertices: list[tuple[float, float]]  # [(x1,y1), (x2,y2), ...]
    zone_type: str = "restricted"  # "restricted", "monitored", "entry", "exit"


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting algorithm to check if point is inside polygon."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def _class_display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def _bbox_center(bbox: dict) -> tuple[float, float]:
    return (bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0


def detect_zone_intrusions(
    tracked_detections: list[dict],
    track_summaries: dict,
    zones: list[Zone],
) -> list[dict]:
    """
    Detect when tracked objects enter restricted zones.

    Returns list of zone_intrusion event dicts.
    """
    events = []
    if not zones:
        return events

    # Group detections by track_id
    track_detections = defaultdict(list)
    for det in tracked_detections:
        if det["track_id"] is not None:
            track_detections[det["track_id"]].append(det)

    for zone in zones:
        if zone.zone_type != "restricted":
            continue

        for track_id, dets in track_detections.items():
            # Check if any detection center falls within this zone
            intrusion_start = None
            intrusion_start_ts = None

            for det in sorted(dets, key=lambda d: d["frame_number"]):
                cx, cy = _bbox_center(det["bbox"])
                in_zone = _point_in_polygon(cx, cy, zone.vertices)

                if in_zone and intrusion_start is None:
                    intrusion_start = det["frame_number"]
                    intrusion_start_ts = det["timestamp_sec"]
                elif not in_zone and intrusion_start is not None:
                    cls_name = det["class_name"]
                    display_name = _class_display_name(cls_name)
                    events.append({
                        "event_id": str(uuid.uuid4()),
                        "event_type": "zone_intrusion",
                        "class_name": cls_name,
                        "track_id": track_id,
                        "start_time_sec": round(intrusion_start_ts, 3),
                        "end_time_sec": round(det["timestamp_sec"], 3),
                        "description": (
                            f"{display_name} (Track #{track_id}) intruded into "
                            f"zone '{zone.name}' from {_format_timestamp(intrusion_start_ts)} "
                            f"to {_format_timestamp(det['timestamp_sec'])}"
                        ),
                        "confidence": 0.85,
                        "metadata": {
                            "zone_id": zone.zone_id,
                            "zone_name": zone.name,
                        },
                    })
                    intrusion_start = None
                    intrusion_start_ts = None

            # Close any open intrusion
            if intrusion_start is not None and dets:
                last_det = max(dets, key=lambda d: d["frame_number"])
                cls_name = last_det["class_name"]
                display_name = _class_display_name(cls_name)
                events.append({
                    "event_id": str(uuid.uuid4()),
                    "event_type": "zone_intrusion",
                    "class_name": cls_name,
                    "track_id": track_id,
                    "start_time_sec": round(intrusion_start_ts, 3),
                    "end_time_sec": round(last_det["timestamp_sec"], 3),
                    "description": (
                        f"{display_name} (Track #{track_id}) intruded into "
                        f"zone '{zone.name}' from {_format_timestamp(intrusion_start_ts)} "
                        f"to {_format_timestamp(last_det['timestamp_sec'])}"
                    ),
                    "confidence": 0.85,
                    "metadata": {
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                    },
                })

    logger.info(f"Zone intrusion detection: {len(events)} intrusions found")
    return events


def detect_loitering_spatial(
    tracked_detections: list[dict],
    track_summaries: dict,
    video_width: int,
    video_height: int,
    time_threshold: float = LOITERING_THRESHOLD_SEC,
    spatial_threshold: float = LOITERING_SPATIAL_THRESH,
) -> list[dict]:
    """
    Enhanced loitering detection using spatial clustering.
    Only triggers if a person stays within a small area for an extended time.

    Args:
        tracked_detections: All detections with track IDs
        track_summaries: Track summary dict
        video_width/height: Video dimensions
        time_threshold: Minimum seconds to consider loitering
        spatial_threshold: Max movement as fraction of frame diagonal

    Returns:
        List of loitering event dicts
    """
    events = []
    frame_diagonal = math.sqrt(video_width ** 2 + video_height ** 2)
    max_movement_px = frame_diagonal * spatial_threshold

    # Group person detections by track_id
    person_tracks = defaultdict(list)
    for det in tracked_detections:
        if det["class_name"] == "person" and det["track_id"] is not None:
            person_tracks[det["track_id"]].append(det)

    for track_id, dets in person_tracks.items():
        if len(dets) < 2:
            continue

        sorted_dets = sorted(dets, key=lambda d: d["timestamp_sec"])
        duration = sorted_dets[-1]["timestamp_sec"] - sorted_dets[0]["timestamp_sec"]

        if duration < time_threshold:
            continue

        # Calculate bounding box of all detection centers
        centers = [_bbox_center(d["bbox"]) for d in sorted_dets]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]

        # Max displacement from centroid
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        max_dist = max(
            math.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2)
            for x, y in centers
        )

        if max_dist <= max_movement_px:
            start_ts = sorted_dets[0]["timestamp_sec"]
            end_ts = sorted_dets[-1]["timestamp_sec"]
            events.append({
                "event_id": str(uuid.uuid4()),
                "event_type": "loitering",
                "class_name": "person",
                "track_id": track_id,
                "start_time_sec": round(start_ts, 3),
                "end_time_sec": round(end_ts, 3),
                "description": (
                    f"Person (Track #{track_id}) loitered in a small area "
                    f"for {duration:.0f} seconds "
                    f"({_format_timestamp(start_ts)} - {_format_timestamp(end_ts)}), "
                    f"max movement: {max_dist:.0f}px"
                ),
                "confidence": min(0.95, 0.6 + (duration / 120.0)),  # Higher conf for longer loiter
                "metadata": {
                    "duration_sec": round(duration, 2),
                    "max_displacement_px": round(max_dist, 1),
                    "centroid": [round(mean_x, 1), round(mean_y, 1)],
                },
            })

    logger.info(f"Spatial loitering detection: {len(events)} loitering events found")
    return events


def detect_object_left_behind(
    tracked_detections: list[dict],
    track_summaries: dict,
    time_threshold: float = OBJECT_LEFT_THRESHOLD_SEC,
    target_classes: set = OBJECT_LEFT_CLASSES,
) -> list[dict]:
    """
    Detect objects (backpacks, suitcases, bags) that appear stationary
    for an extended period — possible abandoned objects.

    Returns list of object_left event dicts.
    """
    events = []

    # Group by track_id, filter to target classes
    obj_tracks = defaultdict(list)
    for det in tracked_detections:
        if det["class_name"] in target_classes and det["track_id"] is not None:
            obj_tracks[det["track_id"]].append(det)

    for track_id, dets in obj_tracks.items():
        if len(dets) < 3:
            continue

        sorted_dets = sorted(dets, key=lambda d: d["timestamp_sec"])
        duration = sorted_dets[-1]["timestamp_sec"] - sorted_dets[0]["timestamp_sec"]

        if duration < time_threshold:
            continue

        # Check if object is mostly stationary
        centers = [_bbox_center(d["bbox"]) for d in sorted_dets]
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        # Stationary if bounding box doesn't move much (< 50px total)
        if x_range < 50 and y_range < 50:
            cls_name = sorted_dets[0]["class_name"]
            display_name = _class_display_name(cls_name)
            start_ts = sorted_dets[0]["timestamp_sec"]
            end_ts = sorted_dets[-1]["timestamp_sec"]

            events.append({
                "event_id": str(uuid.uuid4()),
                "event_type": "object_left",
                "class_name": cls_name,
                "track_id": track_id,
                "start_time_sec": round(start_ts, 3),
                "end_time_sec": round(end_ts, 3),
                "description": (
                    f"Possible abandoned {display_name.lower()} detected (Track #{track_id}) "
                    f"stationary for {duration:.0f} seconds "
                    f"({_format_timestamp(start_ts)} - {_format_timestamp(end_ts)})"
                ),
                "confidence": min(0.9, 0.5 + (duration / 180.0)),
                "metadata": {
                    "duration_sec": round(duration, 2),
                    "position": {
                        "x": round(sum(xs) / len(xs), 1),
                        "y": round(sum(ys) / len(ys), 1),
                    },
                },
            })

    logger.info(f"Object-left detection: {len(events)} abandoned object events found")
    return events


def detect_crowd_density(
    tracked_detections: list[dict],
    zones: list[Zone] | None = None,
    threshold: int = CROWD_THRESHOLD,
) -> list[dict]:
    """
    Enhanced crowd detection — global and per-zone.
    Tracks crowd formation and dispersal as distinct events.

    Returns list of crowd_threshold event dicts.
    """
    events = []

    # Global crowd detection (per-frame person count)
    frame_data = defaultdict(lambda: {"count": 0, "timestamp": 0.0, "positions": []})
    for det in tracked_detections:
        if det["class_name"] == "person":
            fn = det["frame_number"]
            frame_data[fn]["count"] += 1
            frame_data[fn]["timestamp"] = det["timestamp_sec"]
            frame_data[fn]["positions"].append(_bbox_center(det["bbox"]))

    # Find continuous crowd periods
    crowd_start_ts = None
    max_count_in_period = 0

    for frame_num in sorted(frame_data.keys()):
        fd = frame_data[frame_num]
        if fd["count"] >= threshold:
            if crowd_start_ts is None:
                crowd_start_ts = fd["timestamp"]
                max_count_in_period = fd["count"]
            else:
                max_count_in_period = max(max_count_in_period, fd["count"])
        else:
            if crowd_start_ts is not None:
                crowd_end_ts = frame_data[frame_num]["timestamp"]
                duration = crowd_end_ts - crowd_start_ts
                events.append({
                    "event_id": str(uuid.uuid4()),
                    "event_type": "crowd_threshold",
                    "class_name": "person",
                    "track_id": None,
                    "start_time_sec": round(crowd_start_ts, 3),
                    "end_time_sec": round(crowd_end_ts, 3),
                    "description": (
                        f"Crowd of up to {max_count_in_period} people detected "
                        f"from {_format_timestamp(crowd_start_ts)} to "
                        f"{_format_timestamp(crowd_end_ts)} ({duration:.1f}s)"
                    ),
                    "confidence": min(0.95, 0.7 + (max_count_in_period / 20.0)),
                    "metadata": {
                        "max_count": max_count_in_period,
                        "duration_sec": round(duration, 2),
                    },
                })
                crowd_start_ts = None
                max_count_in_period = 0

    # Close any open crowd event
    if crowd_start_ts is not None:
        sorted_frames = sorted(frame_data.keys())
        last_ts = frame_data[sorted_frames[-1]]["timestamp"]
        duration = last_ts - crowd_start_ts
        events.append({
            "event_id": str(uuid.uuid4()),
            "event_type": "crowd_threshold",
            "class_name": "person",
            "track_id": None,
            "start_time_sec": round(crowd_start_ts, 3),
            "end_time_sec": round(last_ts, 3),
            "description": (
                f"Crowd of up to {max_count_in_period} people detected "
                f"from {_format_timestamp(crowd_start_ts)} to "
                f"{_format_timestamp(last_ts)} ({duration:.1f}s)"
            ),
            "confidence": min(0.95, 0.7 + (max_count_in_period / 20.0)),
            "metadata": {
                "max_count": max_count_in_period,
                "duration_sec": round(duration, 2),
            },
        })

    # Per-zone crowd detection
    if zones:
        for zone in zones:
            zone_frame_count = defaultdict(int)
            for det in tracked_detections:
                if det["class_name"] == "person":
                    cx, cy = _bbox_center(det["bbox"])
                    if _point_in_polygon(cx, cy, zone.vertices):
                        zone_frame_count[det["frame_number"]] += 1

            for frame_num, count in zone_frame_count.items():
                if count >= threshold:
                    ts = frame_data[frame_num]["timestamp"]
                    events.append({
                        "event_id": str(uuid.uuid4()),
                        "event_type": "crowd_threshold",
                        "class_name": "person",
                        "track_id": None,
                        "start_time_sec": round(ts, 3),
                        "end_time_sec": round(ts, 3),
                        "description": (
                            f"Crowd of {count} people in zone '{zone.name}' "
                            f"at {_format_timestamp(ts)}"
                        ),
                        "confidence": 0.8,
                        "metadata": {
                            "zone_id": zone.zone_id,
                            "zone_name": zone.name,
                            "count": count,
                        },
                    })

    logger.info(f"Crowd detection: {len(events)} crowd events found")
    return events
