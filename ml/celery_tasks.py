"""
Celery Task Definitions for VaultVision ML Pipeline
The backend creates Celery tasks that call these functions.
"""
import logging
from celery import Celery

from ml.config import REDIS_URL

logger = logging.getLogger(__name__)

# Celery app — shared with backend
celery_app = Celery(
    "vaultvision",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="vaultvision.process_video")
def process_video_task(self, video_path: str, video_id: str, video_filename: str = ""):
    """
    Celery task that runs the ML pipeline on a video.
    Updates task state with progress for WebSocket relay.
    """
    from ml.pipeline.main import process_video

    def progress_callback(pct: float, step: str, message: str):
        self.update_state(
            state="PROCESSING",
            meta={
                "progress_percent": round(pct, 1),
                "current_step": step,
                "message": message,
                "video_id": video_id,
            },
        )

    result = process_video(
        video_path=video_path,
        video_id=video_id,
        video_filename=video_filename,
        progress_callback=progress_callback,
    )

    # Return as dict for JSON serialization
    return result.model_dump(mode="json")


@celery_app.task(name="vaultvision.search_events")
def search_events_task(query: str, video_id: str = None, limit: int = 10):
    """Celery task for NL search (can be called async by backend)."""
    from ml.search.query_engine import QueryEngine
    qe = QueryEngine()
    return qe.search(query=query, video_id=video_id, limit=limit)
