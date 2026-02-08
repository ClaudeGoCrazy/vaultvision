"""
Tripwire / Line-Crossing Detection
Detects when tracked objects cross virtual lines with directional counting.

Features:
- Define virtual lines (tripwires) with start/end points
- Directional crossing detection (A→B vs B→A)
- Per-class counting (people vs vehicles crossing)
- Crossing speed estimation
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Tripwire:
    """A virtual line that objects can cross."""
    wire_id: str
    name: str
    x1: float  # Start point (normalized 0-1)
    y1: float
    x2: float  # End point (normalized 0-1)
    y2: float
    bidirectional: bool = True  # Count both directions
    classes_of_interest: list[str] = field(default_factory=list)  # Empty = all classes


@dataclass
class CrossingEvent:
    """A detected line-crossing event."""
    wire_id: str
    wire_name: str
    track_id: int
    class_name: str
    direction: str  # "A_to_B" or "B_to_A"
    crossing_time_sec: float
    speed_px_per_sec: float
    crossing_point: tuple[float, float]  # Where the crossing occurred


def _line_side(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Determine which side of a line a point is on.
    Returns positive for one side, negative for the other, 0 if on the line.
    Uses cross product of line vector and point vector.
    """
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _segments_intersect(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> tuple[bool, float, float]:
    """
    Check if line segment A (movement path) intersects line segment B (tripwire).
    Returns (intersects, intersection_x, intersection_y).
    """
    dx1 = ax2 - ax1
    dy1 = ay2 - ay1
    dx2 = bx2 - bx1
    dy2 = by2 - by1

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:
        return False, 0.0, 0.0

    t = ((bx1 - ax1) * dy2 - (by1 - ay1) * dx2) / denom
    u = ((bx1 - ax1) * dy1 - (by1 - ay1) * dx1) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = ax1 + t * dx1
        iy = ay1 + t * dy1
        return True, ix, iy

    return False, 0.0, 0.0


def detect_line_crossings(
    tracked_detections: list[dict],
    track_summaries: dict,
    tripwires: list[Tripwire],
    video_width: int,
    video_height: int,
    fps_processed: float = 2.0,
) -> dict:
    """
    Detect when tracked objects cross virtual tripwire lines.

    Args:
        tracked_detections: All detections with track_id, bbox, timestamp
        track_summaries: Track metadata from tracker
        tripwires: List of Tripwire definitions
        video_width: Frame width for denormalizing coords
        video_height: Frame height
        fps_processed: Processing FPS for speed calculation

    Returns:
        {
            "crossings": [CrossingEvent, ...],
            "wire_counts": {
                wire_id: {
                    "a_to_b": int,
                    "b_to_a": int,
                    "total": int,
                    "by_class": {"person": {"a_to_b": n, "b_to_a": n}, ...}
                }
            },
            "events": [dict, ...]  # Event dicts for pipeline integration
        }
    """
    if not tripwires:
        return {"crossings": [], "wire_counts": {}, "events": []}

    # Group detections by track_id, sorted by time
    tracks: dict[int, list[dict]] = {}
    for det in tracked_detections:
        tid = det.get("track_id")
        if tid is not None:
            tracks.setdefault(tid, []).append(det)

    for tid in tracks:
        tracks[tid].sort(key=lambda d: d["timestamp_sec"])

    crossings: list[CrossingEvent] = []
    wire_counts: dict[str, dict] = {}

    # Initialize wire counts
    for wire in tripwires:
        wire_counts[wire.wire_id] = {
            "name": wire.name,
            "a_to_b": 0,
            "b_to_a": 0,
            "total": 0,
            "by_class": {},
        }

    # Check each track against each tripwire
    for tid, dets in tracks.items():
        if len(dets) < 2:
            continue

        class_name = dets[0]["class_name"]

        for wire in tripwires:
            # Filter by class if specified
            if wire.classes_of_interest and class_name not in wire.classes_of_interest:
                continue

            # Denormalize wire coordinates
            wx1 = wire.x1 * video_width
            wy1 = wire.y1 * video_height
            wx2 = wire.x2 * video_width
            wy2 = wire.y2 * video_height

            # Check consecutive detection pairs for crossings
            for i in range(len(dets) - 1):
                d1 = dets[i]
                d2 = dets[i + 1]

                # Get centroids
                b1 = d1["bbox"]
                b2 = d2["bbox"]
                cx1 = (b1["x1"] + b1["x2"]) / 2
                cy1 = (b1["y1"] + b1["y2"]) / 2
                cx2 = (b2["x1"] + b2["x2"]) / 2
                cy2 = (b2["y1"] + b2["y2"]) / 2

                # Check if path crosses the wire
                intersects, ix, iy = _segments_intersect(
                    cx1, cy1, cx2, cy2,
                    wx1, wy1, wx2, wy2,
                )

                if not intersects:
                    continue

                # Determine direction using cross product sign
                side_before = _line_side(cx1, cy1, wx1, wy1, wx2, wy2)
                direction = "A_to_B" if side_before > 0 else "B_to_A"

                if not wire.bidirectional and direction == "B_to_A":
                    continue

                # Calculate crossing speed
                dt = d2["timestamp_sec"] - d1["timestamp_sec"]
                dist = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
                speed = dist / dt if dt > 0 else 0.0

                crossing = CrossingEvent(
                    wire_id=wire.wire_id,
                    wire_name=wire.name,
                    track_id=tid,
                    class_name=class_name,
                    direction=direction,
                    crossing_time_sec=round((d1["timestamp_sec"] + d2["timestamp_sec"]) / 2, 2),
                    speed_px_per_sec=round(speed, 1),
                    crossing_point=(round(ix / video_width, 3), round(iy / video_height, 3)),
                )
                crossings.append(crossing)

                # Update counts
                wc = wire_counts[wire.wire_id]
                if direction == "A_to_B":
                    wc["a_to_b"] += 1
                else:
                    wc["b_to_a"] += 1
                wc["total"] += 1

                cls_counts = wc["by_class"].setdefault(class_name, {"a_to_b": 0, "b_to_a": 0})
                if direction == "A_to_B":
                    cls_counts["a_to_b"] += 1
                else:
                    cls_counts["b_to_a"] += 1

    # Convert crossings to event dicts for pipeline integration
    events = []
    for c in crossings:
        events.append({
            "event_id": f"tripwire_{c.wire_id}_{c.track_id}_{c.crossing_time_sec}",
            "event_type": "line_crossing",
            "class_name": c.class_name,
            "track_id": c.track_id,
            "start_time_sec": c.crossing_time_sec,
            "end_time_sec": c.crossing_time_sec,
            "confidence": 0.95,
            "description": (
                f"{c.class_name} crossed '{c.wire_name}' ({c.direction.replace('_', ' ')}) "
                f"at {c.crossing_time_sec:.1f}s"
            ),
            "metadata": {
                "wire_id": c.wire_id,
                "wire_name": c.wire_name,
                "direction": c.direction,
                "speed_px_per_sec": c.speed_px_per_sec,
                "crossing_point": c.crossing_point,
            },
        })

    logger.info(
        f"Tripwire detection: {len(crossings)} crossings across {len(tripwires)} wires "
        f"from {len(tracks)} tracks"
    )

    return {
        "crossings": [vars(c) for c in crossings],
        "wire_counts": wire_counts,
        "events": events,
    }


def create_default_tripwires() -> list[Tripwire]:
    """Create sensible default tripwires for common surveillance setups."""
    return [
        Tripwire(
            wire_id="entrance",
            name="Main Entrance",
            x1=0.3, y1=0.8,
            x2=0.7, y2=0.8,
            bidirectional=True,
        ),
        Tripwire(
            wire_id="midline",
            name="Center Line",
            x1=0.5, y1=0.0,
            x2=0.5, y2=1.0,
            bidirectional=True,
        ),
    ]
