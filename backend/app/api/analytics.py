from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import Video, Detection, Event

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/analytics/summary")
async def analytics_summary(db: AsyncSession = Depends(get_db)):
    # Total videos
    total_videos = (await db.execute(select(func.count(Video.id)))).scalar() or 0

    # Processed videos
    total_processed = (await db.execute(
        select(func.count(Video.id)).where(Video.status == "completed")
    )).scalar() or 0

    # Currently processing
    total_processing = (await db.execute(
        select(func.count(Video.id)).where(Video.status == "processing")
    )).scalar() or 0

    # Total detections
    total_detections = (await db.execute(select(func.count(Detection.id)))).scalar() or 0

    # Unique persons across all videos
    total_unique_persons = (await db.execute(
        select(func.coalesce(func.sum(Video.unique_persons), 0))
    )).scalar() or 0

    # Unique vehicles across all videos
    total_unique_vehicles = (await db.execute(
        select(func.coalesce(func.sum(Video.unique_vehicles), 0))
    )).scalar() or 0

    # Total events
    total_events = (await db.execute(select(func.count(Event.id)))).scalar() or 0

    # Total processing hours
    total_processing_sec = (await db.execute(
        select(func.coalesce(func.sum(Video.processing_time_sec), 0.0))
    )).scalar() or 0.0

    return {
        "total_videos": total_videos,
        "total_processed": total_processed,
        "total_processing": total_processing,
        "total_detections": total_detections,
        "total_unique_persons": int(total_unique_persons),
        "total_unique_vehicles": int(total_unique_vehicles),
        "total_events": total_events,
        "total_processing_hours": round(total_processing_sec / 3600, 4),
    }
