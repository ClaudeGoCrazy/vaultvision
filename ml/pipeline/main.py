"""
VaultVision ML Pipeline - Main Orchestrator
Entry point: process_video(video_path, video_id) -> PipelineResult

Performance optimizations:
- Model caching (singleton) — no reload between videos
- Memory-efficient frame processing with cleanup
- Lazy ChromaDB initialization
"""
import sys
import time
import logging
import shutil
from pathlib import Path
from collections import Counter

# Add project root to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.schemas import (
    PipelineResult,
    Detection,
    Event,
    BoundingBox,
    HeatmapData,
)
from ml.pipeline.frame_extractor import extract_frames
from ml.pipeline.tracker import MultiObjectTracker
from ml.pipeline.event_generator import generate_events
from ml.analytics.anomaly import Zone
from ml.analytics.heatmap import generate_heatmap
from ml.search.query_engine import QueryEngine
from ml.config import DEFAULT_FPS, MAX_FRAMES_PER_VIDEO

logger = logging.getLogger(__name__)

# Lazy-initialized singleton QueryEngine
_query_engine: QueryEngine | None = None


def _get_query_engine() -> QueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine()
    return _query_engine


def process_video(
    video_path: str,
    video_id: str,
    fps: float = DEFAULT_FPS,
    video_filename: str = "",
    zones: list[Zone] | None = None,
    cleanup_frames: bool = True,
    progress_callback=None,
) -> PipelineResult:
    """
    Full ML pipeline for processing a single video.

    Args:
        video_path: Path to the video file on disk
        video_id: Unique ID assigned by the backend
        fps: Frames per second to extract (default 2fps)
        video_filename: Original filename (for search indexing)
        zones: Optional list of Zone objects for zone-based anomaly detection
        cleanup_frames: If True, delete extracted frames after processing to save disk
        progress_callback: Optional callable(progress_pct: float, step: str, message: str)

    Returns:
        PipelineResult matching shared/schemas.py contract
    """
    start_time = time.time()

    def _progress(pct: float, step: str, message: str):
        if progress_callback:
            progress_callback(pct, step, message)
        logger.info(f"[{step}] {pct:.0f}% - {message}")

    _progress(0, "frame_extraction", "Starting frame extraction...")

    # =========================================
    # Stage 1: Frame Extraction
    # =========================================
    extraction = extract_frames(
        video_path=video_path,
        video_id=video_id,
        fps=fps,
        progress_callback=lambda pct, msg: _progress(pct * 0.15, "frame_extraction", msg),
    )

    # Safety cap for extremely long videos
    frame_paths = extraction.frame_paths[:MAX_FRAMES_PER_VIDEO]
    frame_timestamps = extraction.frame_timestamps[:MAX_FRAMES_PER_VIDEO]
    if len(extraction.frame_paths) > MAX_FRAMES_PER_VIDEO:
        logger.warning(
            f"Video has {len(extraction.frame_paths)} frames, capped at {MAX_FRAMES_PER_VIDEO}"
        )

    _progress(15, "detection", "Starting object detection & tracking...")

    # =========================================
    # Stage 2: Detection + Tracking (cached model)
    # =========================================
    tracker = MultiObjectTracker()
    tracked_detections, track_summaries = tracker.track_video(
        video_path=video_path,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        progress_callback=lambda pct, msg: _progress(15 + pct * 0.45, "tracking", msg),
    )

    _progress(60, "events", "Generating events & anomaly detection...")

    # =========================================
    # Stage 3: Event Generation + Anomaly Detection
    # =========================================
    raw_events = generate_events(
        tracked_detections=tracked_detections,
        track_summaries=track_summaries,
        video_width=extraction.metadata.width,
        video_height=extraction.metadata.height,
        zones=zones,
    )

    _progress(70, "heatmap", "Generating heatmap...")

    # =========================================
    # Stage 4: Heatmap Generation
    # =========================================
    heatmap_data = generate_heatmap(
        detections=tracked_detections,
        video_width=extraction.metadata.width,
        video_height=extraction.metadata.height,
    )

    _progress(80, "indexing", "Indexing events for search...")

    # =========================================
    # Stage 5: ChromaDB Indexing (lazy singleton)
    # =========================================
    try:
        query_engine = _get_query_engine()
        query_engine.index_events(
            events=raw_events,
            video_id=video_id,
            video_filename=video_filename or Path(video_path).name,
        )
    except Exception as e:
        logger.warning(f"ChromaDB indexing failed (non-fatal): {e}")

    _progress(90, "packaging", "Packaging results...")

    # =========================================
    # Stage 6: Build PipelineResult
    # =========================================
    class_counts = Counter()
    unique_person_tracks = set()
    unique_vehicle_tracks = set()
    vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}

    for det in tracked_detections:
        class_counts[det["class_name"]] += 1
        if det["track_id"] is not None:
            if det["class_name"] == "person":
                unique_person_tracks.add(det["track_id"])
            elif det["class_name"] in vehicle_classes:
                unique_vehicle_tracks.add(det["track_id"])

    schema_detections = [
        Detection(
            detection_id=det["detection_id"],
            frame_number=det["frame_number"],
            timestamp_sec=det["timestamp_sec"],
            class_name=det["class_name"],
            confidence=det["confidence"],
            bbox=BoundingBox(**det["bbox"]),
            track_id=det["track_id"],
        )
        for det in tracked_detections
    ]

    schema_events = [
        Event(
            event_id=evt["event_id"],
            event_type=evt["event_type"],
            class_name=evt["class_name"],
            track_id=evt["track_id"],
            start_time_sec=evt["start_time_sec"],
            end_time_sec=evt["end_time_sec"],
            description=evt["description"],
            confidence=evt["confidence"],
            metadata=evt.get("metadata", {}),
        )
        for evt in raw_events
    ]

    schema_heatmap = HeatmapData(**heatmap_data)

    processing_time = time.time() - start_time

    result = PipelineResult(
        video_id=video_id,
        total_frames=len(frame_paths),
        fps_processed=fps,
        processing_time_sec=round(processing_time, 2),
        detections=schema_detections,
        events=schema_events,
        heatmap=schema_heatmap,
        unique_person_count=len(unique_person_tracks),
        unique_vehicle_count=len(unique_vehicle_tracks),
        object_class_counts=dict(class_counts),
    )

    # =========================================
    # Cleanup: Remove extracted frames to save disk
    # =========================================
    if cleanup_frames:
        try:
            frames_dir = frame_paths[0].parent if frame_paths else None
            if frames_dir and frames_dir.exists():
                shutil.rmtree(frames_dir)
                logger.info(f"Cleaned up {len(frame_paths)} extracted frames")
        except Exception as e:
            logger.warning(f"Frame cleanup failed (non-fatal): {e}")

    _progress(100, "complete", f"Pipeline complete in {processing_time:.1f}s")
    logger.info(
        f"Pipeline results: {len(schema_detections)} detections, "
        f"{len(schema_events)} events, "
        f"{len(unique_person_tracks)} unique persons, "
        f"{len(unique_vehicle_tracks)} unique vehicles, "
        f"processed in {processing_time:.2f}s"
    )

    return result
