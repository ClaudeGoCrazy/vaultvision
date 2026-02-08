"""
VaultVision ML Pipeline - Main Orchestrator
Entry point: process_video(video_path, video_id) -> PipelineResult

Full pipeline with 17 stages:
- Frame extraction -> Detection + Tracking -> Events + Anomalies
- Heatmap, Trajectories, Time-series, Thumbnails, Summary
- Tripwire detection, ReID, Attribute extraction
- Safety (smoke/fire), Scene anomaly detection
- Dwell time & queue analytics
- Auto clip extraction, Highlight reel
- ChromaDB indexing, Predictive patterns
- Model caching and frame cleanup
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
from ml.analytics.trajectories import generate_trajectories
from ml.analytics.timeseries import generate_timeseries
from ml.analytics.thumbnails import extract_event_thumbnails, extract_video_thumbnail
from ml.analytics.summary import generate_summary
from ml.analytics.tripwire import Tripwire, detect_line_crossings, create_default_tripwires
from ml.analytics.reid import process_reid
from ml.analytics.attributes import extract_attributes
from ml.analytics.safety import detect_safety_events
from ml.analytics.clip_extractor import extract_all_event_clips, generate_highlight_reel
from ml.analytics.dwell_queue import DwellZone, analyze_dwell_and_queues
from ml.analytics.scene_anomaly import detect_scene_anomalies
from ml.analytics.predictive import analyze_video_patterns
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
    tripwires: list[Tripwire] | None = None,
    dwell_zones: list[DwellZone] | None = None,
    cleanup_frames: bool = True,
    extract_thumbnails: bool = True,
    enable_reid: bool = True,
    enable_attributes: bool = True,
    enable_safety: bool = True,
    enable_scene_anomaly: bool = True,
    enable_clips: bool = True,
    enable_dwell_queue: bool = True,
    enable_tripwires: bool = True,
    enable_predictive: bool = True,
    progress_callback=None,
) -> PipelineResult:
    """
    Full ML pipeline for processing a single video.

    17-stage pipeline producing PipelineResult + extended metadata:
    - trajectories, timeseries, thumbnails, summary
    - tripwire crossings, ReID, attributes, safety alerts
    - scene anomalies, dwell/queue, clips, predictive patterns
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
        progress_callback=lambda pct, msg: _progress(pct * 0.05, "frame_extraction", msg),
    )

    frame_paths = extraction.frame_paths[:MAX_FRAMES_PER_VIDEO]
    frame_timestamps = extraction.frame_timestamps[:MAX_FRAMES_PER_VIDEO]
    if len(extraction.frame_paths) > MAX_FRAMES_PER_VIDEO:
        logger.warning(f"Capped at {MAX_FRAMES_PER_VIDEO} frames")

    _progress(5, "detection", "Starting object detection & tracking...")

    # =========================================
    # Stage 2: Detection + Tracking
    # =========================================
    tracker = MultiObjectTracker()
    tracked_detections, track_summaries = tracker.track_video(
        video_path=video_path,
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        progress_callback=lambda pct, msg: _progress(5 + pct * 0.20, "tracking", msg),
    )

    _progress(25, "events", "Generating events & anomaly detection...")

    # =========================================
    # Stage 3: Events + Anomaly Detection
    # =========================================
    raw_events = generate_events(
        tracked_detections=tracked_detections,
        track_summaries=track_summaries,
        video_width=extraction.metadata.width,
        video_height=extraction.metadata.height,
        zones=zones,
    )

    _progress(30, "heatmap", "Generating heatmap...")

    # =========================================
    # Stage 4: Heatmap
    # =========================================
    heatmap_data = generate_heatmap(
        detections=tracked_detections,
        video_width=extraction.metadata.width,
        video_height=extraction.metadata.height,
    )

    _progress(33, "trajectories", "Computing trajectories & speed...")

    # =========================================
    # Stage 5: Trajectories + Speed/Direction
    # =========================================
    trajectory_data = generate_trajectories(
        tracked_detections=tracked_detections,
        track_summaries=track_summaries,
        video_width=extraction.metadata.width,
        video_height=extraction.metadata.height,
        fps_processed=fps,
    )

    _progress(36, "timeseries", "Building time-series analytics...")

    # =========================================
    # Stage 6: Time-Series Analytics
    # =========================================
    timeseries_data = generate_timeseries(
        tracked_detections=tracked_detections,
        video_duration_sec=extraction.metadata.duration_sec,
    )

    _progress(39, "tripwire", "Checking tripwire crossings...")

    # =========================================
    # Stage 7: Tripwire / Line-Crossing Detection
    # =========================================
    tripwire_data = {"crossings": [], "wire_counts": {}, "events": []}
    if enable_tripwires:
        try:
            wires = tripwires or create_default_tripwires()
            tripwire_data = detect_line_crossings(
                tracked_detections=tracked_detections,
                track_summaries=track_summaries,
                tripwires=wires,
                video_width=extraction.metadata.width,
                video_height=extraction.metadata.height,
                fps_processed=fps,
            )
            raw_events.extend(tripwire_data["events"])
        except Exception as e:
            logger.warning(f"Tripwire detection failed (non-fatal): {e}")

    _progress(43, "reid", "Running re-identification...")

    # =========================================
    # Stage 8: Re-Identification (ReID)
    # =========================================
    reid_data = {}
    if enable_reid and tracked_detections:
        try:
            reid_data = process_reid(
                video_path=video_path,
                video_id=video_id,
                tracked_detections=tracked_detections,
                track_summaries=track_summaries,
            )
        except Exception as e:
            logger.warning(f"ReID failed (non-fatal): {e}")

    _progress(48, "attributes", "Extracting object attributes...")

    # =========================================
    # Stage 9: Object Attribute Extraction
    # =========================================
    attribute_data = {}
    if enable_attributes and tracked_detections:
        try:
            attribute_data = extract_attributes(
                video_path=video_path,
                tracked_detections=tracked_detections,
                track_summaries=track_summaries,
                video_width=extraction.metadata.width,
                video_height=extraction.metadata.height,
            )
        except Exception as e:
            logger.warning(f"Attribute extraction failed (non-fatal): {e}")

    _progress(53, "safety", "Running smoke & fire detection...")

    # =========================================
    # Stage 10: Smoke & Fire Detection
    # =========================================
    safety_data = {"events": [], "summary": {"fire_detected": False, "smoke_detected": False}}
    if enable_safety:
        try:
            safety_data = detect_safety_events(
                video_path=video_path,
                video_id=video_id,
            )
            raw_events.extend(safety_data["events"])
        except Exception as e:
            logger.warning(f"Safety detection failed (non-fatal): {e}")

    _progress(58, "scene_anomaly", "Detecting scene anomalies...")

    # =========================================
    # Stage 11: Scene Anomaly Detection
    # =========================================
    scene_anomaly_data = {"events": [], "summary": {}}
    if enable_scene_anomaly:
        try:
            scene_anomaly_data = detect_scene_anomalies(
                video_path=video_path,
                video_id=video_id,
            )
            raw_events.extend(scene_anomaly_data["events"])
        except Exception as e:
            logger.warning(f"Scene anomaly detection failed (non-fatal): {e}")

    _progress(63, "dwell_queue", "Analyzing dwell times & queues...")

    # =========================================
    # Stage 12: Dwell Time & Queue Analytics
    # =========================================
    dwell_queue_data = {"events": [], "summary": {}}
    if enable_dwell_queue and tracked_detections:
        try:
            dwell_queue_data = analyze_dwell_and_queues(
                tracked_detections=tracked_detections,
                track_summaries=track_summaries,
                video_width=extraction.metadata.width,
                video_height=extraction.metadata.height,
                video_duration_sec=extraction.metadata.duration_sec,
                zones=dwell_zones,
                fps_processed=fps,
            )
            raw_events.extend(dwell_queue_data["events"])
        except Exception as e:
            logger.warning(f"Dwell/queue analysis failed (non-fatal): {e}")

    _progress(68, "thumbnails", "Extracting event thumbnails...")

    # =========================================
    # Stage 13: Thumbnails
    # =========================================
    thumbnail_map = {}
    video_thumbnail = None
    if extract_thumbnails:
        try:
            thumbnail_map = extract_event_thumbnails(
                video_path=video_path,
                video_id=video_id,
                events=raw_events,
                tracked_detections=tracked_detections,
                max_thumbnails=30,
            )
            video_thumbnail = extract_video_thumbnail(video_path, video_id)
        except Exception as e:
            logger.warning(f"Thumbnail extraction failed (non-fatal): {e}")

    _progress(73, "clips", "Extracting event clips...")

    # =========================================
    # Stage 14: Auto Video Clip Extraction
    # =========================================
    clip_map = {}
    highlight_reel_path = None
    if enable_clips:
        try:
            clip_map = extract_all_event_clips(
                video_path=video_path,
                video_id=video_id,
                events=raw_events,
                tracked_detections=tracked_detections,
                max_clips=20,
            )
            if len(raw_events) >= 3:
                highlight_reel_path = generate_highlight_reel(
                    video_path=video_path,
                    video_id=video_id,
                    events=raw_events,
                    tracked_detections=tracked_detections,
                )
        except Exception as e:
            logger.warning(f"Clip extraction failed (non-fatal): {e}")

    _progress(78, "indexing", "Indexing events for search...")

    # =========================================
    # Stage 15: ChromaDB Indexing
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

    _progress(83, "summary", "Generating activity summary...")

    # =========================================
    # Stage 16: Counts + Activity Summary
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

    activity_summary = generate_summary(
        events=raw_events,
        track_summaries=track_summaries,
        class_counts=dict(class_counts),
        video_duration_sec=extraction.metadata.duration_sec,
        unique_persons=len(unique_person_tracks),
        unique_vehicles=len(unique_vehicle_tracks),
        video_filename=video_filename,
    )

    _progress(88, "predictive", "Analyzing activity patterns...")

    # =========================================
    # Stage 17: Predictive Activity Patterns
    # =========================================
    pattern_data = {}
    if enable_predictive:
        try:
            pattern_data = analyze_video_patterns(
                video_id=video_id,
                timeseries_data=timeseries_data,
                events=raw_events,
                source_id=video_filename or "default",
            )
        except Exception as e:
            logger.warning(f"Predictive analysis failed (non-fatal): {e}")

    _progress(92, "packaging", "Packaging results...")

    # =========================================
    # Build PipelineResult
    # =========================================
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

    # Inject thumbnail + clip paths into event metadata
    for evt in raw_events:
        eid = evt["event_id"]
        if eid in thumbnail_map:
            evt.setdefault("metadata", {})["thumbnail_path"] = thumbnail_map[eid]
        crop_key = f"{eid}_crop"
        if crop_key in thumbnail_map:
            evt.setdefault("metadata", {})["thumbnail_crop_path"] = thumbnail_map[crop_key]
        if eid in clip_map:
            evt.setdefault("metadata", {})["clip_path"] = clip_map[eid]

    # Map new event types to schema-valid enums, preserving originals in metadata
    VALID_EVENT_TYPES = {"entry", "exit", "loitering", "zone_intrusion", "crowd_threshold", "object_left", "anomaly"}
    VALID_CLASS_NAMES = {
        "person", "vehicle", "bicycle", "motorcycle", "bus", "truck", "car",
        "dog", "cat", "backpack", "handbag", "suitcase", "cell_phone", "license_plate", "other",
    }

    schema_events = []
    for evt in raw_events:
        raw_event_type = evt["event_type"]
        raw_class_name = evt["class_name"]

        # Map to valid schema values
        schema_event_type = raw_event_type if raw_event_type in VALID_EVENT_TYPES else "anomaly"
        schema_class_name = raw_class_name if raw_class_name in VALID_CLASS_NAMES else "other"

        # Preserve original types in metadata if they were mapped
        meta = evt.get("metadata", {})
        if schema_event_type != raw_event_type:
            meta["original_event_type"] = raw_event_type
        if schema_class_name != raw_class_name:
            meta["original_class_name"] = raw_class_name

        schema_events.append(Event(
            event_id=evt["event_id"],
            event_type=schema_event_type,
            class_name=schema_class_name,
            track_id=evt["track_id"],
            start_time_sec=evt["start_time_sec"],
            end_time_sec=evt["end_time_sec"],
            description=evt["description"],
            confidence=evt["confidence"],
            metadata=meta,
        ))

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
    # Extended metadata (not in shared schema)
    # Backend can access via get_extended_results(result)
    # =========================================
    result._extended = {
        # Core analytics
        "trajectories": trajectory_data,
        "timeseries": timeseries_data,
        "activity_summary": activity_summary,
        # Media
        "video_thumbnail": video_thumbnail,
        "thumbnail_map": thumbnail_map,
        "clip_map": clip_map,
        "highlight_reel": highlight_reel_path,
        # Advanced analytics
        "tripwire": tripwire_data,
        "reid": reid_data,
        "attributes": attribute_data,
        "safety": safety_data,
        "scene_anomaly": scene_anomaly_data,
        "dwell_queue": dwell_queue_data,
        "predictive_patterns": pattern_data,
        # Video info
        "video_metadata": {
            "width": extraction.metadata.width,
            "height": extraction.metadata.height,
            "original_fps": extraction.metadata.original_fps,
            "duration_sec": extraction.metadata.duration_sec,
        },
    }

    # =========================================
    # Cleanup
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

    # Final log
    safety_str = ""
    if safety_data["summary"].get("fire_detected"):
        safety_str += " FIRE DETECTED!"
    if safety_data["summary"].get("smoke_detected"):
        safety_str += " SMOKE DETECTED!"

    logger.info(
        f"Pipeline results: {len(schema_detections)} detections, "
        f"{len(schema_events)} events, "
        f"{len(unique_person_tracks)} persons, "
        f"{len(unique_vehicle_tracks)} vehicles, "
        f"{trajectory_data['summary']['total_tracks']} trajectories, "
        f"{len(tripwire_data['crossings'])} line crossings, "
        f"{reid_data.get('total_identities', 0)} ReID identities, "
        f"{len(clip_map)} clips, "
        f"processed in {processing_time:.2f}s"
        f"{safety_str}"
    )

    return result


def get_extended_results(result: PipelineResult) -> dict | None:
    """Access the extended metadata from a PipelineResult."""
    return getattr(result, "_extended", None)
