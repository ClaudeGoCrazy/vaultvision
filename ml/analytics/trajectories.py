"""
Movement Trail / Trajectory Analysis
Generates path data for each tracked object — positions over time,
speed estimation, and direction classification.
"""
import math
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def _bbox_center(bbox: dict) -> tuple[float, float]:
    return (bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0


def _classify_direction(dx: float, dy: float) -> str:
    """Classify movement direction from displacement vector."""
    if abs(dx) < 5 and abs(dy) < 5:
        return "stationary"
    angle = math.degrees(math.atan2(-dy, dx))  # -dy because y increases downward
    if -22.5 <= angle < 22.5:
        return "right"
    elif 22.5 <= angle < 67.5:
        return "up-right"
    elif 67.5 <= angle < 112.5:
        return "up"
    elif 112.5 <= angle < 157.5:
        return "up-left"
    elif angle >= 157.5 or angle < -157.5:
        return "left"
    elif -157.5 <= angle < -112.5:
        return "down-left"
    elif -112.5 <= angle < -67.5:
        return "down"
    elif -67.5 <= angle < -22.5:
        return "down-right"
    return "unknown"


def _classify_entry_exit_side(
    first_pos: tuple[float, float],
    last_pos: tuple[float, float],
    video_width: int,
    video_height: int,
    edge_margin: float = 0.15,
) -> tuple[str, str]:
    """Classify which side of the frame the object entered/exited from."""
    margin_x = video_width * edge_margin
    margin_y = video_height * edge_margin

    def _side(x, y):
        sides = []
        if x < margin_x:
            sides.append("left")
        elif x > video_width - margin_x:
            sides.append("right")
        if y < margin_y:
            sides.append("top")
        elif y > video_height - margin_y:
            sides.append("bottom")
        return "-".join(sides) if sides else "center"

    entry_side = _side(*first_pos)
    exit_side = _side(*last_pos)
    return entry_side, exit_side


def generate_trajectories(
    tracked_detections: list[dict],
    track_summaries: dict,
    video_width: int,
    video_height: int,
    fps_processed: float = 2.0,
) -> dict:
    """
    Generate trajectory data for all tracked objects.

    Returns:
        {
            "tracks": {
                track_id: {
                    "class_name": str,
                    "path": [{"x": float, "y": float, "t": float, "frame": int}, ...],
                    "speed_avg_px_per_sec": float,
                    "speed_max_px_per_sec": float,
                    "total_distance_px": float,
                    "direction": str,  # primary direction of travel
                    "entry_side": str,  # left, right, top, bottom, center
                    "exit_side": str,
                    "duration_sec": float,
                    "bbox_size_avg": {"w": float, "h": float},
                },
                ...
            },
            "summary": {
                "total_tracks": int,
                "avg_speed_px_per_sec": float,
                "direction_counts": {"left": n, "right": n, ...},
            }
        }
    """
    # Group detections by track_id
    track_dets = defaultdict(list)
    for det in tracked_detections:
        if det["track_id"] is not None:
            track_dets[det["track_id"]].append(det)

    tracks = {}
    direction_counts = defaultdict(int)
    all_speeds = []

    for track_id, dets in track_dets.items():
        sorted_dets = sorted(dets, key=lambda d: d["timestamp_sec"])

        # Build path
        path = []
        widths = []
        heights = []
        for det in sorted_dets:
            cx, cy = _bbox_center(det["bbox"])
            bw = det["bbox"]["x2"] - det["bbox"]["x1"]
            bh = det["bbox"]["y2"] - det["bbox"]["y1"]
            path.append({
                "x": round(cx, 1),
                "y": round(cy, 1),
                "t": det["timestamp_sec"],
                "frame": det["frame_number"],
            })
            widths.append(bw)
            heights.append(bh)

        if len(path) < 2:
            # Single detection — can't compute speed/direction
            tracks[track_id] = {
                "class_name": sorted_dets[0]["class_name"],
                "path": path,
                "speed_avg_px_per_sec": 0.0,
                "speed_max_px_per_sec": 0.0,
                "total_distance_px": 0.0,
                "direction": "stationary",
                "entry_side": "center",
                "exit_side": "center",
                "duration_sec": 0.0,
                "bbox_size_avg": {
                    "w": round(sum(widths) / len(widths), 1),
                    "h": round(sum(heights) / len(heights), 1),
                },
            }
            continue

        # Compute segment speeds and total distance
        total_distance = 0.0
        segment_speeds = []
        for i in range(1, len(path)):
            dx = path[i]["x"] - path[i - 1]["x"]
            dy = path[i]["y"] - path[i - 1]["y"]
            dist = math.sqrt(dx ** 2 + dy ** 2)
            dt = path[i]["t"] - path[i - 1]["t"]
            total_distance += dist
            if dt > 0:
                segment_speeds.append(dist / dt)

        duration = path[-1]["t"] - path[0]["t"]
        avg_speed = total_distance / duration if duration > 0 else 0.0
        max_speed = max(segment_speeds) if segment_speeds else 0.0

        # Primary direction: from first to last position
        total_dx = path[-1]["x"] - path[0]["x"]
        total_dy = path[-1]["y"] - path[0]["y"]
        direction = _classify_direction(total_dx, total_dy)
        direction_counts[direction] += 1

        # Entry/exit sides
        first_pos = (path[0]["x"], path[0]["y"])
        last_pos = (path[-1]["x"], path[-1]["y"])
        entry_side, exit_side = _classify_entry_exit_side(
            first_pos, last_pos, video_width, video_height
        )

        tracks[track_id] = {
            "class_name": sorted_dets[0]["class_name"],
            "path": path,
            "speed_avg_px_per_sec": round(avg_speed, 1),
            "speed_max_px_per_sec": round(max_speed, 1),
            "total_distance_px": round(total_distance, 1),
            "direction": direction,
            "entry_side": entry_side,
            "exit_side": exit_side,
            "duration_sec": round(duration, 2),
            "bbox_size_avg": {
                "w": round(sum(widths) / len(widths), 1),
                "h": round(sum(heights) / len(heights), 1),
            },
        }
        all_speeds.append(avg_speed)

    summary = {
        "total_tracks": len(tracks),
        "avg_speed_px_per_sec": round(sum(all_speeds) / len(all_speeds), 1) if all_speeds else 0.0,
        "direction_counts": dict(direction_counts),
    }

    logger.info(
        f"Trajectories: {len(tracks)} tracks, "
        f"avg speed {summary['avg_speed_px_per_sec']} px/s, "
        f"directions: {dict(direction_counts)}"
    )

    return {"tracks": tracks, "summary": summary}
