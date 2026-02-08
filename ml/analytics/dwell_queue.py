"""
Dwell Time & Queue Analytics
Measures how long objects spend in defined zones and detects queue formations.

Features:
- Per-zone dwell time tracking
- Queue detection (line formations)
- Wait time estimation
- Occupancy over time
- Congestion alerts
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DwellZone:
    """A zone for measuring dwell time."""
    zone_id: str
    name: str
    # Polygon vertices (normalized 0-1)
    polygon: list[tuple[float, float]]
    max_dwell_sec: float = 300.0  # Alert if someone stays longer than this


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _detect_queue_formation(
    positions: list[tuple[float, float, int]],
    min_queue_size: int = 3,
    max_gap: float = 0.08,
    alignment_threshold: float = 0.05,
) -> list[dict]:
    """
    Detect if people are standing in a line (queue formation).

    Approach: Check if multiple people are roughly collinear.

    Args:
        positions: List of (cx, cy, track_id) normalized coordinates
        min_queue_size: Minimum people to consider a queue
        max_gap: Max gap between consecutive queue members
        alignment_threshold: How close to a line they must be

    Returns:
        List of detected queues with members and line info
    """
    if len(positions) < min_queue_size:
        return []

    queues = []

    # Try to find collinear groups
    # Sort by x, then try horizontal queues
    by_x = sorted(positions, key=lambda p: p[0])
    by_y = sorted(positions, key=lambda p: p[1])

    for sorted_pts, axis_name in [(by_x, "horizontal"), (by_y, "vertical")]:
        # Sliding window to find aligned groups
        for start in range(len(sorted_pts)):
            queue_members = [sorted_pts[start]]

            for j in range(start + 1, len(sorted_pts)):
                p = sorted_pts[j]
                last = queue_members[-1]

                # Check alignment (perpendicular deviation)
                if axis_name == "horizontal":
                    gap = abs(p[0] - last[0])
                    deviation = abs(p[1] - last[1])
                else:
                    gap = abs(p[1] - last[1])
                    deviation = abs(p[0] - last[0])

                if gap <= max_gap and deviation <= alignment_threshold:
                    queue_members.append(p)

            if len(queue_members) >= min_queue_size:
                # Calculate queue line
                xs = [p[0] for p in queue_members]
                ys = [p[1] for p in queue_members]

                queues.append({
                    "members": [p[2] for p in queue_members],  # track_ids
                    "size": len(queue_members),
                    "orientation": axis_name,
                    "start": (round(min(xs), 3), round(min(ys), 3)),
                    "end": (round(max(xs), 3), round(max(ys), 3)),
                    "length": round(math.sqrt(
                        (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2
                    ), 3),
                })

    # Deduplicate overlapping queues
    unique_queues = []
    seen_member_sets = []
    for q in sorted(queues, key=lambda q: q["size"], reverse=True):
        members = set(q["members"])
        is_subset = False
        for seen in seen_member_sets:
            if members <= seen:
                is_subset = True
                break
        if not is_subset:
            unique_queues.append(q)
            seen_member_sets.append(members)

    return unique_queues


def analyze_dwell_and_queues(
    tracked_detections: list[dict],
    track_summaries: dict,
    video_width: int,
    video_height: int,
    video_duration_sec: float,
    zones: list[DwellZone] | None = None,
    fps_processed: float = 2.0,
) -> dict:
    """
    Analyze dwell times and detect queues.

    Args:
        tracked_detections: All detections with track_id, bbox, timestamp
        track_summaries: Track metadata
        video_width: Frame width
        video_height: Frame height
        video_duration_sec: Total video duration
        zones: Optional dwell zones (if None, uses full frame)
        fps_processed: Processing FPS

    Returns:
        {
            "dwell_times": {
                zone_id: {
                    track_id: {
                        "total_dwell_sec": float,
                        "first_seen_sec": float,
                        "last_seen_sec": float,
                        "class_name": str,
                    }
                }
            },
            "zone_occupancy": {
                zone_id: {
                    "timestamps": [t, ...],
                    "counts": [n, ...],
                    "avg_occupancy": float,
                    "peak_occupancy": int,
                    "peak_time_sec": float,
                }
            },
            "queues": [queue_dicts],
            "dwell_alerts": [event_dicts],
            "queue_events": [event_dicts],
            "events": [combined pipeline events],
            "summary": {...}
        }
    """
    # Default to full-frame zone if none specified
    if not zones:
        zones = [DwellZone(
            zone_id="full_frame",
            name="Full Scene",
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
        )]

    # Group detections by track
    tracks: dict[int, list[dict]] = {}
    for det in tracked_detections:
        tid = det.get("track_id")
        if tid is not None:
            tracks.setdefault(tid, []).append(det)

    for tid in tracks:
        tracks[tid].sort(key=lambda d: d["timestamp_sec"])

    # Dwell time calculation
    dwell_times: dict[str, dict] = {z.zone_id: {} for z in zones}
    zone_frames: dict[str, dict[float, set]] = {z.zone_id: defaultdict(set) for z in zones}

    for tid, dets in tracks.items():
        class_name = dets[0]["class_name"]

        for det in dets:
            bbox = det["bbox"]
            cx = ((bbox["x1"] + bbox["x2"]) / 2) / video_width
            cy = ((bbox["y1"] + bbox["y2"]) / 2) / video_height
            ts = det["timestamp_sec"]

            for zone in zones:
                if _point_in_polygon(cx, cy, zone.polygon):
                    # Track dwell time
                    if tid not in dwell_times[zone.zone_id]:
                        dwell_times[zone.zone_id][tid] = {
                            "first_seen_sec": ts,
                            "last_seen_sec": ts,
                            "class_name": class_name,
                            "detection_count": 0,
                        }
                    dwell_times[zone.zone_id][tid]["last_seen_sec"] = ts
                    dwell_times[zone.zone_id][tid]["detection_count"] += 1

                    # Occupancy tracking (per-second buckets)
                    bucket = round(ts, 0)
                    zone_frames[zone.zone_id][bucket].add(tid)

    # Calculate total dwell times
    for zone_id in dwell_times:
        for tid in dwell_times[zone_id]:
            entry = dwell_times[zone_id][tid]
            entry["total_dwell_sec"] = round(
                entry["last_seen_sec"] - entry["first_seen_sec"], 2
            )

    # Occupancy time series
    zone_occupancy = {}
    for zone in zones:
        zid = zone.zone_id
        if not zone_frames[zid]:
            zone_occupancy[zid] = {
                "name": zone.name,
                "timestamps": [],
                "counts": [],
                "avg_occupancy": 0,
                "peak_occupancy": 0,
                "peak_time_sec": 0,
            }
            continue

        timestamps = sorted(zone_frames[zid].keys())
        counts = [len(zone_frames[zid][t]) for t in timestamps]

        peak_idx = counts.index(max(counts)) if counts else 0
        zone_occupancy[zid] = {
            "name": zone.name,
            "timestamps": timestamps,
            "counts": counts,
            "avg_occupancy": round(sum(counts) / len(counts), 1) if counts else 0,
            "peak_occupancy": max(counts) if counts else 0,
            "peak_time_sec": timestamps[peak_idx] if timestamps else 0,
        }

    # Queue detection at each time bucket
    all_queues = []
    interval = 1.0 / fps_processed
    t = 0.0
    while t <= video_duration_sec:
        # Get positions of all people at this timestamp
        positions = []
        for tid, dets in tracks.items():
            if dets[0]["class_name"] != "person":
                continue
            # Find closest detection to this time
            closest = min(dets, key=lambda d: abs(d["timestamp_sec"] - t))
            if abs(closest["timestamp_sec"] - t) < interval * 2:
                bbox = closest["bbox"]
                cx = ((bbox["x1"] + bbox["x2"]) / 2) / video_width
                cy = ((bbox["y1"] + bbox["y2"]) / 2) / video_height
                positions.append((cx, cy, tid))

        queues = _detect_queue_formation(positions)
        for q in queues:
            q["timestamp_sec"] = round(t, 2)
            all_queues.append(q)

        t += 1.0  # Check every second

    # Deduplicate queue detections (merge consecutive similar queues)
    merged_queues = []
    if all_queues:
        current = all_queues[0].copy()
        current["first_seen_sec"] = current["timestamp_sec"]
        current["last_seen_sec"] = current["timestamp_sec"]

        for q in all_queues[1:]:
            if set(q["members"]) & set(current["members"]) and q["timestamp_sec"] - current["last_seen_sec"] < 3:
                current["last_seen_sec"] = q["timestamp_sec"]
                current["size"] = max(current["size"], q["size"])
                current["members"] = list(set(current["members"]) | set(q["members"]))
            else:
                merged_queues.append(current)
                current = q.copy()
                current["first_seen_sec"] = current["timestamp_sec"]
                current["last_seen_sec"] = current["timestamp_sec"]
        merged_queues.append(current)

    # Generate events
    events = []

    # Dwell alerts
    for zone in zones:
        zid = zone.zone_id
        for tid, info in dwell_times[zid].items():
            if info["total_dwell_sec"] >= zone.max_dwell_sec:
                events.append({
                    "event_id": f"dwell_{zid}_{tid}",
                    "event_type": "excessive_dwell",
                    "class_name": info["class_name"],
                    "track_id": tid,
                    "start_time_sec": info["first_seen_sec"],
                    "end_time_sec": info["last_seen_sec"],
                    "confidence": 0.85,
                    "description": (
                        f"{info['class_name']} (track {tid}) remained in '{zone.name}' "
                        f"for {info['total_dwell_sec']:.0f}s (limit: {zone.max_dwell_sec:.0f}s)"
                    ),
                    "metadata": {
                        "zone_id": zid,
                        "zone_name": zone.name,
                        "dwell_sec": info["total_dwell_sec"],
                        "threshold_sec": zone.max_dwell_sec,
                    },
                })

    # Queue events
    for i, q in enumerate(merged_queues):
        duration = q["last_seen_sec"] - q["first_seen_sec"]
        events.append({
            "event_id": f"queue_{i}_{q['first_seen_sec']}",
            "event_type": "queue_detected",
            "class_name": "person",
            "track_id": None,
            "start_time_sec": q["first_seen_sec"],
            "end_time_sec": q["last_seen_sec"],
            "confidence": 0.80,
            "description": (
                f"Queue of {q['size']} people detected ({q['orientation']}) "
                f"from {q['first_seen_sec']:.1f}s to {q['last_seen_sec']:.1f}s"
            ),
            "metadata": {
                "queue_size": q["size"],
                "orientation": q["orientation"],
                "duration_sec": round(duration, 2),
                "member_track_ids": q["members"],
            },
        })

    # Summary
    all_dwell_times = []
    for zid in dwell_times:
        for tid, info in dwell_times[zid].items():
            all_dwell_times.append(info["total_dwell_sec"])

    summary = {
        "total_zones_monitored": len(zones),
        "total_tracks_analyzed": sum(len(dwell_times[z.zone_id]) for z in zones),
        "avg_dwell_sec": round(
            sum(all_dwell_times) / len(all_dwell_times), 1
        ) if all_dwell_times else 0,
        "max_dwell_sec": round(max(all_dwell_times), 1) if all_dwell_times else 0,
        "queues_detected": len(merged_queues),
        "max_queue_size": max((q["size"] for q in merged_queues), default=0),
        "dwell_alerts": sum(1 for e in events if e["event_type"] == "excessive_dwell"),
    }

    result = {
        "dwell_times": {zid: dict(dt) for zid, dt in dwell_times.items()},
        "zone_occupancy": zone_occupancy,
        "queues": merged_queues,
        "events": events,
        "summary": summary,
    }

    logger.info(
        f"Dwell/Queue analysis: {summary['total_tracks_analyzed']} tracks, "
        f"avg dwell {summary['avg_dwell_sec']}s, "
        f"{summary['queues_detected']} queues detected"
    )

    return result
