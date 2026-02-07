"""Media utilities: video duration extraction and thumbnail generation via ffmpeg."""
import json
import logging
import subprocess
import shutil
from pathlib import Path

from app.core.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

THUMBNAILS_DIR = UPLOAD_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)


def _ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def get_video_duration(video_path: str) -> float | None:
    """Extract video duration in seconds using ffprobe. Returns None if unavailable."""
    if not _ffprobe_available():
        logger.warning("ffprobe not found — skipping duration extraction")
        return None

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = data.get("format", {}).get("duration")
            if duration is not None:
                return float(duration)
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")

    return None


def generate_thumbnail(video_path: str, video_id: str) -> str | None:
    """Extract a frame at ~1s as a JPEG thumbnail. Returns relative URL path or None."""
    if not _ffmpeg_available():
        logger.warning("ffmpeg not found — skipping thumbnail generation")
        return None

    thumb_filename = f"{video_id}.jpg"
    thumb_path = THUMBNAILS_DIR / thumb_filename

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-ss", "1",
                "-vframes", "1",
                "-q:v", "3",
                "-vf", "scale=320:-1",
                str(thumb_path),
            ],
            capture_output=True,
            timeout=30,
        )
        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            return f"/uploads/thumbnails/{thumb_filename}"
    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {e}")

    return None
