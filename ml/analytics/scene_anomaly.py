"""
Scene Anomaly Detection (AI-Powered)
Detects unusual or abnormal scenes without predefined rules.

Approach:
- Build a "normal scene" baseline from frame features
- Flag frames that deviate significantly from the baseline
- Uses lightweight CNN features (color histograms + edge density + optical flow)
- No CLIP dependency — works on any machine

Features:
- Adaptive baseline learning from initial frames
- Anomaly scoring (0-1) for each frame
- Temporal smoothing to reduce false positives
- Scene change detection
- Lighting change detection
"""
import logging
import math
from collections import deque
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _extract_scene_features(frame: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """
    Extract a compact scene feature vector from a frame.
    Uses a spatial grid of color + edge features.

    Returns:
        1D feature vector (float32)
    """
    h, w = frame.shape[:2]
    features = []

    # Resize to standard for consistency
    std_frame = cv2.resize(frame, (256, 192))
    std_h, std_w = std_frame.shape[:2]
    cell_h = std_h // grid_size
    cell_w = std_w // grid_size

    hsv = cv2.cvtColor(std_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(std_frame, cv2.COLOR_BGR2GRAY)

    for gy in range(grid_size):
        for gx in range(grid_size):
            y1 = gy * cell_h
            y2 = (gy + 1) * cell_h
            x1 = gx * cell_w
            x2 = (gx + 1) * cell_w

            cell_hsv = hsv[y1:y2, x1:x2]
            cell_gray = gray[y1:y2, x1:x2]

            # Color features: mean + std of H, S, V
            for ch in range(3):
                channel = cell_hsv[:, :, ch].astype(np.float32)
                features.append(np.mean(channel))
                features.append(np.std(channel))

            # Edge density
            edges = cv2.Canny(cell_gray, 50, 150)
            edge_density = np.sum(edges > 0) / (cell_h * cell_w)
            features.append(edge_density)

            # Brightness
            features.append(np.mean(cell_gray))

    # Global features
    features.append(np.mean(gray))  # Overall brightness
    features.append(np.std(gray))   # Contrast
    features.append(cv2.Laplacian(gray, cv2.CV_64F).var())  # Sharpness/blur

    vec = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _compute_optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


class SceneAnomalyDetector:
    """
    Detects unusual scenes by comparing against a learned baseline.
    """

    def __init__(
        self,
        baseline_frames: int = 30,
        anomaly_threshold: float = 0.15,
        smoothing_window: int = 5,
    ):
        """
        Args:
            baseline_frames: Number of initial frames to build baseline
            anomaly_threshold: Feature distance threshold for anomaly (0-1)
            smoothing_window: Temporal smoothing window size
        """
        self.baseline_frames = baseline_frames
        self.anomaly_threshold = anomaly_threshold
        self.smoothing_window = smoothing_window

        # Baseline statistics
        self._baseline_features: list[np.ndarray] = []
        self._baseline_mean: Optional[np.ndarray] = None
        self._baseline_std: Optional[np.ndarray] = None
        self._baseline_ready = False

        # Motion baseline
        self._motion_values: list[float] = []
        self._motion_mean: float = 0.0
        self._motion_std: float = 1.0

        # Smoothing
        self._score_history: deque = deque(maxlen=smoothing_window)

    def _update_baseline(self, feature: np.ndarray):
        """Add a frame to the baseline."""
        self._baseline_features.append(feature)

        if len(self._baseline_features) >= self.baseline_frames:
            feats = np.array(self._baseline_features)
            self._baseline_mean = np.mean(feats, axis=0)
            self._baseline_std = np.std(feats, axis=0) + 1e-6
            self._baseline_ready = True
            logger.info(
                f"Scene baseline established from {len(self._baseline_features)} frames"
            )

    def _update_motion_baseline(self, motion: float):
        """Update motion baseline statistics."""
        self._motion_values.append(motion)
        if len(self._motion_values) >= self.baseline_frames:
            self._motion_mean = sum(self._motion_values) / len(self._motion_values)
            self._motion_std = math.sqrt(
                sum((m - self._motion_mean) ** 2 for m in self._motion_values) /
                len(self._motion_values)
            ) + 1e-6

    def score_frame(self, feature: np.ndarray, motion: float = 0.0) -> float:
        """
        Compute anomaly score for a frame.

        Returns:
            Score 0-1 (0 = normal, 1 = highly anomalous)
        """
        if not self._baseline_ready:
            self._update_baseline(feature)
            self._update_motion_baseline(motion)
            return 0.0

        # Feature-based anomaly score (Mahalanobis-like distance)
        z_scores = np.abs((feature - self._baseline_mean) / self._baseline_std)
        feature_score = float(np.mean(z_scores))

        # Motion-based anomaly score
        motion_z = abs(motion - self._motion_mean) / self._motion_std if self._motion_std > 0 else 0
        motion_score = min(motion_z / 5.0, 1.0)  # Normalize

        # Combined score (weighted)
        raw_score = 0.7 * min(feature_score / 3.0, 1.0) + 0.3 * motion_score

        # Temporal smoothing
        self._score_history.append(raw_score)
        smoothed = sum(self._score_history) / len(self._score_history)

        return round(min(smoothed, 1.0), 3)


def detect_scene_anomalies(
    video_path: str,
    video_id: str,
    fps_sample: float = 1.0,
    anomaly_threshold: float = 0.15,
    baseline_sec: float = 10.0,
) -> dict:
    """
    Analyze a video for scene-level anomalies.

    Args:
        video_path: Path to video file
        video_id: Video identifier
        fps_sample: Frames per second to analyze
        anomaly_threshold: Anomaly score threshold (0-1)
        baseline_sec: Seconds of video to use as baseline

    Returns:
        {
            "anomaly_scores": [(timestamp, score), ...],
            "anomalous_frames": [(timestamp, score, description), ...],
            "scene_changes": [(timestamp, magnitude), ...],
            "lighting_changes": [(timestamp, brightness_delta), ...],
            "events": [pipeline event dicts],
            "summary": {
                "total_frames_analyzed": int,
                "anomalous_frame_count": int,
                "anomaly_percentage": float,
                "max_anomaly_score": float,
                "scene_change_count": int,
                "avg_anomaly_score": float,
            }
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video for scene analysis: {video_path}")
        return _empty_result()

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps_sample))

    baseline_frames = max(10, int(baseline_sec * fps_sample))
    detector = SceneAnomalyDetector(
        baseline_frames=baseline_frames,
        anomaly_threshold=anomaly_threshold,
    )

    anomaly_scores = []
    anomalous_frames = []
    scene_changes = []
    lighting_changes = []

    prev_gray = None
    prev_brightness = None
    frame_num = 0
    analyzed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval != 0:
            frame_num += 1
            continue

        timestamp = frame_num / video_fps

        # Extract features
        feature = _extract_scene_features(frame)
        gray = cv2.cvtColor(cv2.resize(frame, (256, 192)), cv2.COLOR_BGR2GRAY)

        # Optical flow (motion)
        motion = 0.0
        if prev_gray is not None:
            motion = _compute_optical_flow_magnitude(prev_gray, gray)

        # Anomaly score
        score = detector.score_frame(feature, motion)
        anomaly_scores.append((round(timestamp, 2), score))

        if score >= anomaly_threshold and analyzed > baseline_frames:
            description = _describe_anomaly(score, motion, detector)
            anomalous_frames.append((round(timestamp, 2), score, description))

        # Scene change detection (large feature shift)
        if prev_gray is not None:
            hist_curr = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
            hist_prev = cv2.calcHist([prev_gray], [0], None, [64], [0, 256]).flatten()
            hist_curr = hist_curr / (hist_curr.sum() + 1e-8)
            hist_prev = hist_prev / (hist_prev.sum() + 1e-8)
            hist_diff = cv2.compareHist(
                hist_curr.astype(np.float32),
                hist_prev.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )

            if hist_diff > 0.5:
                scene_changes.append((round(timestamp, 2), round(float(hist_diff), 3)))

        # Lighting change detection
        brightness = float(np.mean(gray))
        if prev_brightness is not None:
            brightness_delta = abs(brightness - prev_brightness)
            if brightness_delta > 30:
                lighting_changes.append((round(timestamp, 2), round(brightness_delta, 1)))

        prev_gray = gray.copy()
        prev_brightness = brightness
        frame_num += 1
        analyzed += 1

    cap.release()

    # Generate pipeline events
    events = []

    # Anomalous periods (merge consecutive anomalous frames)
    if anomalous_frames:
        period_start = anomalous_frames[0][0]
        period_max_score = anomalous_frames[0][1]

        for i in range(1, len(anomalous_frames)):
            curr_time = anomalous_frames[i][0]
            prev_time = anomalous_frames[i - 1][0]

            if curr_time - prev_time > 3.0:  # Gap > 3s = new period
                events.append({
                    "event_id": f"scene_anomaly_{video_id}_{len(events)}",
                    "event_type": "scene_anomaly",
                    "class_name": "scene",
                    "track_id": None,
                    "start_time_sec": period_start,
                    "end_time_sec": prev_time,
                    "confidence": round(min(period_max_score * 2, 0.95), 2),
                    "description": (
                        f"Unusual scene activity detected from {period_start:.1f}s "
                        f"to {prev_time:.1f}s (anomaly score: {period_max_score:.2f})"
                    ),
                    "metadata": {
                        "max_anomaly_score": period_max_score,
                        "severity": "high" if period_max_score > 0.5 else "medium",
                    },
                })
                period_start = curr_time
                period_max_score = anomalous_frames[i][1]
            else:
                period_max_score = max(period_max_score, anomalous_frames[i][1])

        # Close final period
        events.append({
            "event_id": f"scene_anomaly_{video_id}_{len(events)}",
            "event_type": "scene_anomaly",
            "class_name": "scene",
            "track_id": None,
            "start_time_sec": period_start,
            "end_time_sec": anomalous_frames[-1][0],
            "confidence": round(min(period_max_score * 2, 0.95), 2),
            "description": (
                f"Unusual scene activity detected from {period_start:.1f}s "
                f"to {anomalous_frames[-1][0]:.1f}s (score: {period_max_score:.2f})"
            ),
            "metadata": {
                "max_anomaly_score": period_max_score,
                "severity": "high" if period_max_score > 0.5 else "medium",
            },
        })

    # Scene change events
    for ts, mag in scene_changes:
        events.append({
            "event_id": f"scene_change_{video_id}_{ts}",
            "event_type": "scene_change",
            "class_name": "scene",
            "track_id": None,
            "start_time_sec": ts,
            "end_time_sec": ts,
            "confidence": round(min(mag, 0.95), 2),
            "description": f"Scene change detected at {ts:.1f}s (magnitude: {mag:.2f})",
            "metadata": {"magnitude": mag},
        })

    # Summary
    all_scores = [s for _, s in anomaly_scores]
    summary = {
        "total_frames_analyzed": analyzed,
        "anomalous_frame_count": len(anomalous_frames),
        "anomaly_percentage": round(
            len(anomalous_frames) / analyzed * 100, 1
        ) if analyzed > 0 else 0,
        "max_anomaly_score": round(max(all_scores), 3) if all_scores else 0,
        "avg_anomaly_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
        "scene_change_count": len(scene_changes),
        "lighting_change_count": len(lighting_changes),
    }

    logger.info(
        f"Scene analysis: {analyzed} frames, "
        f"{len(anomalous_frames)} anomalous ({summary['anomaly_percentage']}%), "
        f"{len(scene_changes)} scene changes"
    )

    return {
        "anomaly_scores": anomaly_scores,
        "anomalous_frames": anomalous_frames,
        "scene_changes": scene_changes,
        "lighting_changes": lighting_changes,
        "events": events,
        "summary": summary,
    }


def _describe_anomaly(score: float, motion: float, detector: SceneAnomalyDetector) -> str:
    """Generate a human-readable anomaly description."""
    parts = []

    if score > 0.5:
        parts.append("Highly unusual scene content")
    elif score > 0.3:
        parts.append("Moderately unusual scene")
    else:
        parts.append("Slightly unusual activity")

    if motion > detector._motion_mean + 2 * detector._motion_std:
        parts.append("with abnormal motion")
    elif motion < detector._motion_mean - detector._motion_std:
        parts.append("with unusually low motion")

    return ", ".join(parts)


def _empty_result() -> dict:
    return {
        "anomaly_scores": [],
        "anomalous_frames": [],
        "scene_changes": [],
        "lighting_changes": [],
        "events": [],
        "summary": {
            "total_frames_analyzed": 0,
            "anomalous_frame_count": 0,
            "anomaly_percentage": 0,
            "max_anomaly_score": 0,
            "avg_anomaly_score": 0,
            "scene_change_count": 0,
            "lighting_change_count": 0,
        },
    }
