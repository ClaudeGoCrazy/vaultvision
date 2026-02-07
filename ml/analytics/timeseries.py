"""
Time-Series Analytics
Generates per-interval data for dashboard charts:
- Detection counts over time
- People count over time
- Vehicle count over time
- Activity heatmap by time-of-day (if video has real timestamps)
"""
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}


def generate_timeseries(
    tracked_detections: list[dict],
    video_duration_sec: float,
    interval_sec: float = 1.0,
) -> dict:
    """
    Generate time-series analytics data.

    Args:
        tracked_detections: All detections with timestamps
        video_duration_sec: Total video duration
        interval_sec: Time bucket size in seconds

    Returns:
        {
            "interval_sec": float,
            "timestamps": [0.0, 1.0, 2.0, ...],
            "total_detections": [n, n, n, ...],
            "person_count": [n, n, n, ...],
            "vehicle_count": [n, n, n, ...],
            "unique_person_count": [n, n, n, ...],  # unique track IDs per interval
            "unique_vehicle_count": [n, n, n, ...],
            "class_breakdown": {
                "person": [n, n, ...],
                "car": [n, n, ...],
                ...
            },
            "activity_score": [0-1, ...],  # normalized activity level per interval
            "peak_activity_time_sec": float,
            "quietest_time_sec": float,
        }
    """
    num_buckets = max(1, math.ceil(video_duration_sec / interval_sec))

    timestamps = [round(i * interval_sec, 2) for i in range(num_buckets)]
    total_dets = [0] * num_buckets
    person_count = [0] * num_buckets
    vehicle_count = [0] * num_buckets
    unique_persons = [set() for _ in range(num_buckets)]
    unique_vehicles = [set() for _ in range(num_buckets)]
    class_buckets = defaultdict(lambda: [0] * num_buckets)

    for det in tracked_detections:
        ts = det["timestamp_sec"]
        bucket = min(int(ts / interval_sec), num_buckets - 1)

        total_dets[bucket] += 1
        class_buckets[det["class_name"]][bucket] += 1

        if det["class_name"] == "person":
            person_count[bucket] += 1
            if det["track_id"] is not None:
                unique_persons[bucket].add(det["track_id"])
        elif det["class_name"] in VEHICLE_CLASSES:
            vehicle_count[bucket] += 1
            if det["track_id"] is not None:
                unique_vehicles[bucket].add(det["track_id"])

    unique_person_counts = [len(s) for s in unique_persons]
    unique_vehicle_counts = [len(s) for s in unique_vehicles]

    # Activity score: normalized total detections (0-1)
    max_dets = max(total_dets) if total_dets else 1
    activity_score = [round(d / max_dets, 3) if max_dets > 0 else 0 for d in total_dets]

    # Peak and quietest
    peak_idx = total_dets.index(max(total_dets)) if total_dets else 0
    # Find quietest non-zero period, or first bucket if all zero
    nonzero = [(i, d) for i, d in enumerate(total_dets) if d > 0]
    if nonzero:
        quietest_idx = min(nonzero, key=lambda x: x[1])[0]
    else:
        quietest_idx = 0

    result = {
        "interval_sec": interval_sec,
        "timestamps": timestamps,
        "total_detections": total_dets,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "unique_person_count": unique_person_counts,
        "unique_vehicle_count": unique_vehicle_counts,
        "class_breakdown": {k: v for k, v in class_buckets.items()},
        "activity_score": activity_score,
        "peak_activity_time_sec": round(timestamps[peak_idx], 2),
        "quietest_time_sec": round(timestamps[quietest_idx], 2),
    }

    logger.info(
        f"Time-series: {num_buckets} buckets @ {interval_sec}s, "
        f"peak activity at {result['peak_activity_time_sec']}s"
    )

    return result
