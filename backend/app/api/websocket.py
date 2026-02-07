"""WebSocket endpoint for real-time processing progress."""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from app.core.database import async_session
from app.models.models import Video

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, video_id: str, websocket: WebSocket):
        await websocket.accept()
        if video_id not in self.active_connections:
            self.active_connections[video_id] = []
        self.active_connections[video_id].append(websocket)

    def disconnect(self, video_id: str, websocket: WebSocket):
        if video_id in self.active_connections:
            self.active_connections[video_id].remove(websocket)
            if not self.active_connections[video_id]:
                del self.active_connections[video_id]

    async def broadcast(self, video_id: str, message: dict):
        if video_id in self.active_connections:
            for ws in self.active_connections[video_id]:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws/progress/{video_id}")
async def websocket_progress(websocket: WebSocket, video_id: str):
    await manager.connect(video_id, websocket)
    try:
        while True:
            # Poll DB for status updates and push to client
            async with async_session() as db:
                video = await db.get(Video, video_id)
                if video:
                    msg = {
                        "type": "progress",
                        "video_id": video.id,
                        "status": video.status,
                        "progress_percent": video.progress_percent,
                        "current_step": video.current_step or "",
                        "message": None,
                    }
                    await websocket.send_json(msg)

                    if video.status in ("completed", "failed"):
                        await websocket.send_json({
                            "type": "progress",
                            "video_id": video.id,
                            "status": video.status,
                            "progress_percent": 100.0 if video.status == "completed" else video.progress_percent,
                            "current_step": "done" if video.status == "completed" else "failed",
                            "message": video.error_message,
                        })
                        break

            # Also listen for client messages (like ping/disconnect)
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(video_id, websocket)
