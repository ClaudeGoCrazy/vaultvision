"""Service to store ML pipeline results into the database."""
import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from app.models.models import Video, Detection, Event, Heatmap


async def store_pipeline_result(db: AsyncSession, result: dict):
    """Store a PipelineResult dict (matching shared/schemas.py) into the database."""
    video_id = result["video_id"]
    video = await db.get(Video, video_id)
    if not video:
        raise ValueError(f"Video {video_id} not found")

    # Update video metadata
    video.status = "completed"
    video.progress_percent = 100.0
    video.current_step = "completed"
    video.total_frames = result.get("total_frames", 0)
    video.fps_processed = result.get("fps_processed", 0.0)
    video.processing_time_sec = result.get("processing_time_sec", 0.0)
    video.unique_persons = result.get("unique_person_count", 0)
    video.unique_vehicles = result.get("unique_vehicle_count", 0)
    video.total_detections = len(result.get("detections", []))
    video.completed_time = datetime.now(timezone.utc)

    # Store detections
    for det in result.get("detections", []):
        bbox = det.get("bbox", {})
        detection = Detection(
            video_id=video_id,
            detection_id=det["detection_id"],
            frame_number=det["frame_number"],
            timestamp_sec=det["timestamp_sec"],
            class_name=det["class_name"],
            confidence=det["confidence"],
            bbox_x1=bbox.get("x1", 0),
            bbox_y1=bbox.get("y1", 0),
            bbox_x2=bbox.get("x2", 0),
            bbox_y2=bbox.get("y2", 0),
            track_id=det.get("track_id"),
        )
        db.add(detection)

    # Store events
    for evt in result.get("events", []):
        event = Event(
            video_id=video_id,
            event_id=evt["event_id"],
            event_type=evt["event_type"],
            class_name=evt["class_name"],
            track_id=evt.get("track_id"),
            start_time_sec=evt["start_time_sec"],
            end_time_sec=evt.get("end_time_sec"),
            description=evt["description"],
            confidence=evt["confidence"],
            metadata_json=evt.get("metadata", {}),
        )
        db.add(event)

    # Store heatmap
    heatmap_data = result.get("heatmap")
    if heatmap_data:
        # Remove old heatmap if exists
        old = await db.execute(select(Heatmap).where(Heatmap.video_id == video_id))
        old_heatmap = old.scalar_one_or_none()
        if old_heatmap:
            await db.delete(old_heatmap)

        heatmap = Heatmap(
            video_id=video_id,
            width=heatmap_data["width"],
            height=heatmap_data["height"],
            grid_json=heatmap_data["grid"],
            video_width=heatmap_data["video_width"],
            video_height=heatmap_data["video_height"],
        )
        db.add(heatmap)

    await db.commit()
