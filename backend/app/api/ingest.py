"""Endpoint to receive ML pipeline results directly (for when Celery isn't used)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.auth import verify_api_key
from app.services.pipeline import store_pipeline_result

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post("/ingest/pipeline-result")
async def ingest_pipeline_result(
    result: dict,
    db: AsyncSession = Depends(get_db),
    _key: str = Depends(verify_api_key),
):
    """Accept a PipelineResult JSON from the ML pipeline and store it.

    This is an alternative to Celery — the ML pipeline can POST results
    directly to this endpoint after processing a video.
    """
    video_id = result.get("video_id")
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")

    try:
        await store_pipeline_result(db, result)
        return {"status": "stored", "video_id": video_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
