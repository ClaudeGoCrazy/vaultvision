"""
Frame Extraction Pipeline
Extracts frames from video at configurable FPS using OpenCV.
"""
import cv2
import logging
from pathlib import Path
from dataclasses import dataclass

from ml.config import DEFAULT_FPS, FRAMES_DIR

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""
    width: int
    height: int
    total_frames: int
    original_fps: float
    duration_sec: float


@dataclass
class ExtractionResult:
    """Result of frame extraction."""
    frame_paths: list[Path]
    frame_timestamps: list[float]  # timestamp in seconds for each frame
    metadata: VideoMetadata


def extract_frames(
    video_path: str,
    video_id: str,
    fps: float = DEFAULT_FPS,
    progress_callback=None,
) -> ExtractionResult:
    """
    Extract frames from a video at the specified FPS rate.

    Args:
        video_path: Path to the input video file
        video_id: Unique ID for organizing output
        fps: Frames per second to extract (default 2fps)
        progress_callback: Optional callable(progress_pct, message) for progress updates

    Returns:
        ExtractionResult with frame paths, timestamps, and video metadata
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Extract video metadata
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / original_fps if original_fps > 0 else 0.0

    metadata = VideoMetadata(
        width=width,
        height=height,
        total_frames=total_frames,
        original_fps=original_fps,
        duration_sec=duration_sec,
    )

    logger.info(
        f"Video: {video_path.name} | {width}x{height} | "
        f"{original_fps:.1f}fps | {total_frames} frames | {duration_sec:.1f}s"
    )

    # Create output directory for this video's frames
    output_dir = FRAMES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate frame interval
    frame_interval = original_fps / fps if fps > 0 else 1
    frame_paths = []
    frame_timestamps = []
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at the desired interval
        if frame_idx % max(1, int(frame_interval)) == 0:
            timestamp_sec = frame_idx / original_fps if original_fps > 0 else 0.0
            frame_filename = f"frame_{extracted_count:06d}.jpg"
            frame_path = output_dir / frame_filename

            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(frame_path)
            frame_timestamps.append(timestamp_sec)
            extracted_count += 1

        frame_idx += 1

        # Progress updates every 100 frames
        if progress_callback and frame_idx % 100 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            progress_callback(pct, f"Extracting frames: {extracted_count} extracted")

    cap.release()
    logger.info(f"Extracted {extracted_count} frames at {fps}fps from {total_frames} total frames")

    return ExtractionResult(
        frame_paths=frame_paths,
        frame_timestamps=frame_timestamps,
        metadata=metadata,
    )
