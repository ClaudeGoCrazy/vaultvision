"""
Cross-Camera Correlation
Links events and tracks across multiple camera feeds using:
- Appearance similarity (ReID embeddings)
- Temporal proximity
- Spatial topology (camera adjacency graph)

Features:
- Multi-camera identity tracking
- Transition time estimation between cameras
- Path reconstruction across camera network
- Camera handoff detection
"""
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraNode:
    """Represents a camera in the network."""
    camera_id: str
    name: str
    location: tuple[float, float] = (0.0, 0.0)  # lat/lon or relative position
    adjacent_cameras: list[str] = field(default_factory=list)
    # Expected transition times to adjacent cameras (seconds)
    transition_times: dict[str, tuple[float, float]] = field(default_factory=dict)  # cam_id -> (min_sec, max_sec)


@dataclass
class CrossCameraMatch:
    """A detected identity match across cameras."""
    global_id: str
    camera_a: str
    camera_b: str
    track_id_a: int
    track_id_b: int
    similarity: float
    time_a: float  # Last seen in camera A
    time_b: float  # First seen in camera B
    transition_time_sec: float
    class_name: str


class CrossCameraCorrelator:
    """
    Correlates identities and events across multiple camera feeds.
    """

    def __init__(self, cameras: list[CameraNode] | None = None):
        self.cameras: dict[str, CameraNode] = {}
        if cameras:
            for cam in cameras:
                self.cameras[cam.camera_id] = cam

        # Per-camera track data: camera_id -> {track_id: track_info}
        self.camera_tracks: dict[str, dict[int, dict]] = defaultdict(dict)
        # Global identity matches
        self.matches: list[CrossCameraMatch] = []
        # Identity paths: global_id -> [(camera_id, enter_time, exit_time)]
        self.identity_paths: dict[str, list[dict]] = defaultdict(list)

    def add_camera(self, camera: CameraNode):
        """Register a camera in the network."""
        self.cameras[camera.camera_id] = camera

    def register_tracks(
        self,
        camera_id: str,
        video_id: str,
        tracked_detections: list[dict],
        track_summaries: dict,
        reid_data: dict | None = None,
    ):
        """
        Register tracks from a camera feed.

        Args:
            camera_id: Camera identifier
            video_id: Video identifier
            tracked_detections: Detections with track_id, bbox, timestamp
            track_summaries: Track metadata
            reid_data: ReID results if available (track_to_global mapping)
        """
        tracks: dict[int, list[dict]] = {}
        for det in tracked_detections:
            tid = det.get("track_id")
            if tid is not None:
                tracks.setdefault(tid, []).append(det)

        for tid, dets in tracks.items():
            sorted_dets = sorted(dets, key=lambda d: d["timestamp_sec"])
            first_det = sorted_dets[0]
            last_det = sorted_dets[-1]

            # Compute average centroid for rough position
            centroids = []
            for d in sorted_dets:
                bbox = d["bbox"]
                cx = (bbox["x1"] + bbox["x2"]) / 2
                cy = (bbox["y1"] + bbox["y2"]) / 2
                centroids.append((cx, cy))

            track_info = {
                "track_id": tid,
                "video_id": video_id,
                "camera_id": camera_id,
                "class_name": first_det["class_name"],
                "first_seen_sec": first_det["timestamp_sec"],
                "last_seen_sec": last_det["timestamp_sec"],
                "duration_sec": last_det["timestamp_sec"] - first_det["timestamp_sec"],
                "avg_centroid": (
                    sum(c[0] for c in centroids) / len(centroids),
                    sum(c[1] for c in centroids) / len(centroids),
                ),
                "exit_side": self._classify_exit_side(last_det["bbox"]),
                "entry_side": self._classify_exit_side(first_det["bbox"]),
                "global_id": reid_data.get("track_to_global", {}).get(tid) if reid_data else None,
            }

            self.camera_tracks[camera_id][tid] = track_info

        logger.info(
            f"Cross-camera: Registered {len(tracks)} tracks from camera '{camera_id}'"
        )

    def _classify_exit_side(self, bbox: dict, margin: float = 50) -> str:
        """Classify which side of the frame a detection is near."""
        cx = (bbox["x1"] + bbox["x2"]) / 2
        cy = (bbox["y1"] + bbox["y2"]) / 2

        # Simple heuristic — check proximity to frame edges
        if bbox["x1"] < margin:
            return "left"
        if bbox["y1"] < margin:
            return "top"
        if bbox["x2"] > 600:  # Approximate — would need frame dims
            return "right"
        if bbox["y2"] > 400:
            return "bottom"
        return "center"

    def correlate(
        self,
        similarity_threshold: float = 0.70,
        max_transition_sec: float = 300.0,
    ) -> dict:
        """
        Find identity matches across all registered cameras.

        Args:
            similarity_threshold: Min similarity for ReID matching
            max_transition_sec: Max time gap between cameras

        Returns:
            {
                "matches": [CrossCameraMatch dicts],
                "identity_paths": {global_id: [camera appearances]},
                "transitions": [{from_cam, to_cam, track_ids, transition_time}],
                "summary": {...}
            }
        """
        matches = []
        transitions = []

        camera_ids = list(self.camera_tracks.keys())

        for i in range(len(camera_ids)):
            for j in range(i + 1, len(camera_ids)):
                cam_a = camera_ids[i]
                cam_b = camera_ids[j]

                # Check each track pair across cameras
                for tid_a, info_a in self.camera_tracks[cam_a].items():
                    for tid_b, info_b in self.camera_tracks[cam_b].items():
                        # Must be same class
                        if info_a["class_name"] != info_b["class_name"]:
                            continue

                        # Check temporal feasibility
                        # Track A exits before Track B enters
                        transition_ab = info_b["first_seen_sec"] - info_a["last_seen_sec"]
                        transition_ba = info_a["first_seen_sec"] - info_b["last_seen_sec"]

                        feasible_transition = None
                        if 0 < transition_ab <= max_transition_sec:
                            feasible_transition = transition_ab
                            time_a = info_a["last_seen_sec"]
                            time_b = info_b["first_seen_sec"]
                        elif 0 < transition_ba <= max_transition_sec:
                            feasible_transition = transition_ba
                            time_a = info_b["last_seen_sec"]
                            time_b = info_a["first_seen_sec"]
                        else:
                            continue

                        # Check camera adjacency if available
                        cam_a_node = self.cameras.get(cam_a)
                        if cam_a_node and cam_a_node.adjacent_cameras:
                            if cam_b not in cam_a_node.adjacent_cameras:
                                continue
                            # Check against expected transition times
                            if cam_b in cam_a_node.transition_times:
                                min_t, max_t = cam_a_node.transition_times[cam_b]
                                if not (min_t <= feasible_transition <= max_t):
                                    continue

                        # Match by global ID if both have ReID
                        similarity = 0.0
                        if info_a["global_id"] and info_b["global_id"]:
                            if info_a["global_id"] == info_b["global_id"]:
                                similarity = 0.95
                            else:
                                continue  # Different global IDs, no match
                        else:
                            # Heuristic matching based on exit/entry sides and timing
                            score = 0.5  # Base

                            # Temporal proximity bonus
                            if feasible_transition < 30:
                                score += 0.2
                            elif feasible_transition < 120:
                                score += 0.1

                            # Exit/entry side consistency bonus
                            if (info_a["exit_side"] in ("right", "bottom") and
                                    info_b["entry_side"] in ("left", "top")):
                                score += 0.15
                            elif (info_a["exit_side"] in ("left", "top") and
                                  info_b["entry_side"] in ("right", "bottom")):
                                score += 0.15

                            similarity = min(score, 0.95)

                        if similarity >= similarity_threshold:
                            global_id = (
                                info_a["global_id"] or
                                info_b["global_id"] or
                                f"XMATCH_{cam_a}_{tid_a}_{cam_b}_{tid_b}"
                            )

                            match = CrossCameraMatch(
                                global_id=global_id,
                                camera_a=cam_a,
                                camera_b=cam_b,
                                track_id_a=tid_a,
                                track_id_b=tid_b,
                                similarity=round(similarity, 3),
                                time_a=time_a,
                                time_b=time_b,
                                transition_time_sec=round(feasible_transition, 2),
                                class_name=info_a["class_name"],
                            )
                            matches.append(match)

                            transitions.append({
                                "from_camera": cam_a,
                                "to_camera": cam_b,
                                "track_id_from": tid_a,
                                "track_id_to": tid_b,
                                "global_id": global_id,
                                "transition_time_sec": round(feasible_transition, 2),
                                "class_name": info_a["class_name"],
                            })

                            # Build identity path
                            self.identity_paths[global_id].append({
                                "camera_id": cam_a,
                                "track_id": tid_a,
                                "enter_sec": info_a["first_seen_sec"],
                                "exit_sec": info_a["last_seen_sec"],
                            })
                            self.identity_paths[global_id].append({
                                "camera_id": cam_b,
                                "track_id": tid_b,
                                "enter_sec": info_b["first_seen_sec"],
                                "exit_sec": info_b["last_seen_sec"],
                            })

        # Deduplicate identity paths
        for gid in self.identity_paths:
            seen = set()
            deduped = []
            for entry in self.identity_paths[gid]:
                key = (entry["camera_id"], entry["track_id"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(entry)
            self.identity_paths[gid] = sorted(deduped, key=lambda e: e["enter_sec"])

        summary = {
            "cameras_analyzed": len(camera_ids),
            "total_tracks": sum(len(t) for t in self.camera_tracks.values()),
            "matches_found": len(matches),
            "unique_cross_camera_identities": len(self.identity_paths),
            "avg_transition_time_sec": (
                round(sum(m.transition_time_sec for m in matches) / len(matches), 1)
                if matches else 0
            ),
        }

        logger.info(
            f"Cross-camera correlation: {summary['matches_found']} matches "
            f"across {summary['cameras_analyzed']} cameras"
        )

        return {
            "matches": [vars(m) for m in matches],
            "identity_paths": dict(self.identity_paths),
            "transitions": transitions,
            "summary": summary,
        }


# Singleton correlator
_correlator: Optional[CrossCameraCorrelator] = None


def get_cross_camera_correlator() -> CrossCameraCorrelator:
    global _correlator
    if _correlator is None:
        _correlator = CrossCameraCorrelator()
    return _correlator
