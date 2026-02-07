import os
import uuid
import shutil
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Header, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import UPLOAD_DIR, ALLOWED_VIDEO_EXTENSIONS
from app.core.auth import verify_api_key
from app.models.models import Video, Detection, Event, Heatmap
from app.services.media import get_video_duration, generate_thumbnail

router = APIRouter(prefix="/api/v1", tags=["videos"])


# ── POST /api/v1/videos/upload ──────────────────────────────────
@router.post("/videos/upload")
async def upload_video(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    video_id = str(uuid.uuid4())
    safe_filename = f"{video_id}{ext}"
    file_path = UPLOAD_DIR / safe_filename

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    # Extract duration and thumbnail (non-blocking — gracefully skipped if ffmpeg missing)
    duration = get_video_duration(str(file_path))
    thumbnail = generate_thumbnail(str(file_path), video_id)

    video = Video(
        id=video_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_path=str(file_path),
        status="pending",
        duration_sec=duration,
        thumbnail_path=thumbnail,
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    # Kick off Celery task (import here to avoid circular at module level)
    try:
        from app.services.tasks import process_video_task
        process_video_task.delay(str(file_path), video_id)
    except Exception:
        pass  # Celery may not be running; video stays in "pending"

    return {
        "video_id": video_id,
        "filename": file.filename,
        "status": video.status,
        "message": "Video uploaded successfully. Processing queued.",
    }


# ── GET /api/v1/videos ──────────────────────────────────────────
@router.get("/videos")
async def list_videos(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Video).order_by(Video.upload_time.desc()))
    videos = result.scalars().all()
    return [
        {
            "video_id": v.id,
            "filename": v.original_filename,
            "status": v.status,
            "duration_sec": v.duration_sec,
            "upload_time": v.upload_time.isoformat() if v.upload_time else None,
            "total_detections": v.total_detections,
            "unique_persons": v.unique_persons,
            "unique_vehicles": v.unique_vehicles,
            "thumbnail_path": v.thumbnail_path,
        }
        for v in videos
    ]


# ── GET /api/v1/videos/{id}/status ──────────────────────────────
@router.get("/videos/{video_id}/status")
async def get_video_status(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return {
        "video_id": video.id,
        "status": video.status,
        "progress_percent": video.progress_percent,
        "current_step": video.current_step,
        "estimated_remaining_sec": None,
    }


# ── GET /api/v1/videos/{id}/detections ──────────────────────────
@router.get("/videos/{video_id}/detections")
async def get_detections(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(
        select(Detection)
        .where(Detection.video_id == video_id)
        .order_by(Detection.frame_number)
    )
    detections = result.scalars().all()
    return [
        {
            "detection_id": d.detection_id,
            "frame_number": d.frame_number,
            "timestamp_sec": d.timestamp_sec,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox": {
                "x1": d.bbox_x1,
                "y1": d.bbox_y1,
                "x2": d.bbox_x2,
                "y2": d.bbox_y2,
            },
            "track_id": d.track_id,
        }
        for d in detections
    ]


# ── GET /api/v1/videos/{id}/heatmap ─────────────────────────────
@router.get("/videos/{video_id}/heatmap")
async def get_heatmap(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(
        select(Heatmap).where(Heatmap.video_id == video_id)
    )
    heatmap = result.scalar_one_or_none()
    if not heatmap:
        raise HTTPException(status_code=404, detail="Heatmap data not available yet")

    return {
        "width": heatmap.width,
        "height": heatmap.height,
        "grid": heatmap.grid_json,
        "video_width": heatmap.video_width,
        "video_height": heatmap.video_height,
    }


# ── DELETE /api/v1/videos/{id} ──────────────────────────────────
@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: str,
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete video file from disk
    video_path = Path(video.file_path)
    if video_path.exists():
        video_path.unlink()

    await db.delete(video)
    await db.commit()
    return {"message": f"Video {video_id} deleted successfully"}


# ── GET /api/v1/videos/{id}/events ──────────────────────────────
@router.get("/videos/{video_id}/events")
async def get_events(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(
        select(Event)
        .where(Event.video_id == video_id)
        .order_by(Event.start_time_sec)
    )
    events = result.scalars().all()
    return [
        {
            "event_id": e.event_id,
            "event_type": e.event_type,
            "class_name": e.class_name,
            "track_id": e.track_id,
            "start_time_sec": e.start_time_sec,
            "end_time_sec": e.end_time_sec,
            "description": e.description,
            "confidence": e.confidence,
            "metadata": e.metadata_json or {},
        }
        for e in events
    ]


# ── GET /api/v1/videos/{id}/stream ──────────────────────────────
@router.get("/videos/{video_id}/stream")
async def stream_video(
    video_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = Path(video.file_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    file_size = video_path.stat().st_size
    content_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"

    range_header = request.headers.get("range")
    if range_header:
        # Parse Range: bytes=START-END
        range_val = range_header.strip().split("=")[1]
        range_parts = range_val.split("-")
        start = int(range_parts[0])
        end = int(range_parts[1]) if range_parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_range():
            with open(video_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=content_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    # No range — stream the whole file
    def iter_file():
        with open(video_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk

    return StreamingResponse(
        iter_file(),
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )
