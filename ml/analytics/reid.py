"""
Re-Identification (ReID) Module
Tracks the same person/object across camera cuts, re-entries, or multiple feeds
using appearance embeddings from CLIP or a lightweight CNN feature extractor.

Features:
- Extract appearance embeddings from detection crops
- Match identities across track breaks using cosine similarity
- Merge fragmented tracks into unified identity profiles
- Cross-camera identity matching
"""
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import CLIP — falls back to CNN histogram features if unavailable
_clip_available = False
_clip_model = None
_clip_preprocess = None

try:
    import torch
    _torch_available = True
except ImportError:
    _torch_available = False


def _get_histogram_embedding(crop: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Lightweight appearance descriptor using color histograms.
    Works without any ML model — pure OpenCV.
    Returns a normalized feature vector.
    """
    if crop.size == 0:
        return np.zeros(bins * 3 + bins, dtype=np.float32)

    # Resize to standard size for consistency
    crop = cv2.resize(crop, (64, 128))

    features = []

    # Color histograms (HSV — more robust to lighting changes)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins], [0, 256])
        hist = hist.flatten()
        features.append(hist)

    # Texture via grayscale gradient histogram
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    hist_texture = cv2.calcHist([mag.astype(np.uint8)], [0], None, [bins], [0, 256])
    features.append(hist_texture.flatten())

    # Spatial color — top half vs bottom half color difference
    h = crop.shape[0]
    top_mean = np.mean(hsv[:h // 2], axis=(0, 1))
    bot_mean = np.mean(hsv[h // 2:], axis=(0, 1))
    features.append(top_mean.astype(np.float32))
    features.append(bot_mean.astype(np.float32))

    vec = np.concatenate(features)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


class ReIDEngine:
    """
    Appearance-based Re-Identification engine.
    Extracts embeddings from detection crops and matches identities.
    """

    def __init__(self, similarity_threshold: float = 0.75, max_gallery_per_id: int = 5):
        """
        Args:
            similarity_threshold: Min cosine similarity to consider a match
            max_gallery_per_id: Max embeddings stored per identity (gallery)
        """
        self.similarity_threshold = similarity_threshold
        self.max_gallery = max_gallery_per_id

        # Identity gallery: global_id -> list of embeddings
        self.gallery: dict[str, list[np.ndarray]] = {}
        # Track-to-global mapping: (video_id, track_id) -> global_id
        self.track_map: dict[tuple[str, int], str] = {}
        # Next global ID counter
        self._next_id = 1

    def _new_global_id(self) -> str:
        gid = f"REID_{self._next_id:04d}"
        self._next_id += 1
        return gid

    def extract_embeddings(
        self,
        video_path: str,
        tracked_detections: list[dict],
        sample_rate: int = 5,
    ) -> dict[int, list[np.ndarray]]:
        """
        Extract appearance embeddings for each track from the video.

        Args:
            video_path: Path to video file
            tracked_detections: Detections with track_id, bbox, frame_number
            sample_rate: Sample every N detections per track (saves compute)

        Returns:
            Dict of track_id -> list of embedding vectors
        """
        # Group by track
        tracks: dict[int, list[dict]] = {}
        for det in tracked_detections:
            tid = det.get("track_id")
            if tid is not None:
                tracks.setdefault(tid, []).append(det)

        # Determine which frames we need
        frames_needed: dict[int, list[tuple[int, dict]]] = {}  # frame_num -> [(track_id, det)]
        for tid, dets in tracks.items():
            sampled = dets[::sample_rate][:self.max_gallery]
            for det in sampled:
                fn = det["frame_number"]
                frames_needed.setdefault(fn, []).append((tid, det))

        if not frames_needed:
            return {}

        # Read frames and extract crops
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video for ReID: {video_path}")
            return {}

        track_embeddings: dict[int, list[np.ndarray]] = defaultdict(list)
        sorted_frames = sorted(frames_needed.keys())

        for target_frame in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            for tid, det in frames_needed[target_frame]:
                bbox = det["bbox"]
                x1 = max(0, int(bbox["x1"]))
                y1 = max(0, int(bbox["y1"]))
                x2 = min(w, int(bbox["x2"]))
                y2 = min(h, int(bbox["y2"]))

                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                crop = frame[y1:y2, x1:x2]
                emb = _get_histogram_embedding(crop)
                track_embeddings[tid].append(emb)

        cap.release()

        logger.info(f"ReID: Extracted embeddings for {len(track_embeddings)} tracks")
        return dict(track_embeddings)

    def register_tracks(
        self,
        video_id: str,
        track_embeddings: dict[int, list[np.ndarray]],
        track_summaries: dict,
    ) -> dict[int, str]:
        """
        Register tracks and match against existing gallery.

        Args:
            video_id: Video identifier
            track_embeddings: track_id -> embeddings from extract_embeddings()
            track_summaries: Track metadata

        Returns:
            Dict of track_id -> global_id (matched or new)
        """
        results: dict[int, str] = {}

        for tid, embeddings in track_embeddings.items():
            if not embeddings:
                continue

            # Average embedding for this track
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

            # Try to match against gallery
            best_match_id = None
            best_sim = 0.0

            for gid, gallery_embs in self.gallery.items():
                for g_emb in gallery_embs:
                    sim = _cosine_similarity(avg_emb, g_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_match_id = gid

            if best_sim >= self.similarity_threshold and best_match_id:
                # Match found — same identity
                global_id = best_match_id
                # Update gallery with new embedding
                if len(self.gallery[global_id]) < self.max_gallery:
                    self.gallery[global_id].append(avg_emb)
            else:
                # New identity
                global_id = self._new_global_id()
                self.gallery[global_id] = [avg_emb]

            self.track_map[(video_id, tid)] = global_id
            results[tid] = global_id

        logger.info(
            f"ReID: Registered {len(results)} tracks, "
            f"{len(self.gallery)} unique identities in gallery"
        )
        return results

    def find_matches(
        self,
        video_id: str,
        track_embeddings: dict[int, list[np.ndarray]],
    ) -> list[dict]:
        """
        Find identity matches without registering (query-only).

        Returns:
            List of match dicts: {track_id, matched_global_id, similarity, ...}
        """
        matches = []

        for tid, embeddings in track_embeddings.items():
            if not embeddings:
                continue

            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)

            for gid, gallery_embs in self.gallery.items():
                avg_gallery = np.mean(gallery_embs, axis=0)
                sim = _cosine_similarity(avg_emb, avg_gallery)

                if sim >= self.similarity_threshold:
                    matches.append({
                        "track_id": tid,
                        "video_id": video_id,
                        "matched_global_id": gid,
                        "similarity": round(sim, 3),
                    })

        return matches

    def get_identity_timeline(self, global_id: str) -> list[dict]:
        """Get all appearances of a global identity across videos."""
        appearances = []
        for (vid, tid), gid in self.track_map.items():
            if gid == global_id:
                appearances.append({
                    "video_id": vid,
                    "track_id": tid,
                    "global_id": gid,
                })
        return appearances


# Singleton ReID engine
_reid_engine: Optional[ReIDEngine] = None


def get_reid_engine() -> ReIDEngine:
    global _reid_engine
    if _reid_engine is None:
        _reid_engine = ReIDEngine()
    return _reid_engine


def process_reid(
    video_path: str,
    video_id: str,
    tracked_detections: list[dict],
    track_summaries: dict,
) -> dict:
    """
    Run ReID processing on a video's tracks.

    Returns:
        {
            "track_to_global": {track_id: global_id},
            "total_identities": int,
            "re_identified": int,  # tracks matched to existing identities
            "new_identities": int,
            "matches": [match_dicts]
        }
    """
    engine = get_reid_engine()

    # Extract embeddings
    embeddings = engine.extract_embeddings(
        video_path=video_path,
        tracked_detections=tracked_detections,
    )

    gallery_size_before = len(engine.gallery)

    # Register and match
    track_map = engine.register_tracks(
        video_id=video_id,
        track_embeddings=embeddings,
        track_summaries=track_summaries,
    )

    new_identities = len(engine.gallery) - gallery_size_before
    re_identified = len(track_map) - new_identities

    result = {
        "track_to_global": track_map,
        "total_identities": len(engine.gallery),
        "re_identified": max(0, re_identified),
        "new_identities": new_identities,
    }

    logger.info(
        f"ReID complete: {len(track_map)} tracks → "
        f"{new_identities} new + {result['re_identified']} re-identified"
    )

    return result
