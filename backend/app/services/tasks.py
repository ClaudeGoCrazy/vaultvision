"""Celery tasks for async video processing."""
import asyncio
import json
from celery import Celery
from app.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

celery_app = Celery(
    "vaultvision",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


def _run_async(coro):
    """Helper to run async code from sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, name="process_video")
def process_video_task(self, video_path: str, video_id: str):
    """Process a video through the ML pipeline and store results.

    The ML engineer's pipeline is called here. For now, we update
    status to 'processing' and wait for the ML pipeline to be wired in.
    """
    from app.core.database import async_session
    from app.models.models import Video

    async def _update_status(status: str, progress: float = 0.0, step: str = ""):
        async with async_session() as db:
            video = await db.get(Video, video_id)
            if video:
                video.status = status
                video.progress_percent = progress
                video.current_step = step
                await db.commit()

    # Mark as processing
    _run_async(_update_status("processing", 0.0, "Initializing pipeline"))

    try:
        # Try to import and run the ML pipeline
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent))
        from ml.pipeline.main import process_video

        _run_async(_update_status("processing", 10.0, "Running ML pipeline"))

        # Run the ML pipeline
        result = process_video(video_path, video_id)

        _run_async(_update_status("processing", 90.0, "Storing results"))

        # Store results in DB
        from app.services.pipeline import store_pipeline_result

        async def _store():
            async with async_session() as db:
                if isinstance(result, str):
                    result_dict = json.loads(result)
                elif hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                else:
                    result_dict = result
                await store_pipeline_result(db, result_dict)

        _run_async(_store())

        return {"status": "completed", "video_id": video_id}

    except ImportError:
        # ML pipeline not available yet — mark as pending for reprocessing
        _run_async(_update_status("pending", 0.0, "ML pipeline not available"))
        return {"status": "pending", "video_id": video_id, "error": "ML pipeline not installed"}

    except Exception as e:
        _run_async(_update_status("failed", 0.0, str(e)[:200]))
        return {"status": "failed", "video_id": video_id, "error": str(e)}
