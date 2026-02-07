"""
Heatmap Generator
Accumulates detection bounding box centers into a 2D density matrix.
Output matches HeatmapData schema from shared/schemas.py.
"""
import logging
import numpy as np

from ml.config import HEATMAP_GRID_WIDTH, HEATMAP_GRID_HEIGHT

logger = logging.getLogger(__name__)


def generate_heatmap(
    detections: list[dict],
    video_width: int,
    video_height: int,
    grid_width: int = HEATMAP_GRID_WIDTH,
    grid_height: int = HEATMAP_GRID_HEIGHT,
    class_filter: str | None = None,
) -> dict:
    """
    Generate a 2D density heatmap from detection bounding boxes.

    Args:
        detections: List of detection dicts with bbox info
        video_width: Original video pixel width
        video_height: Original video pixel height
        grid_width: Number of columns in the heatmap grid
        grid_height: Number of rows in the heatmap grid
        class_filter: Optional - only include detections of this class

    Returns:
        Dict matching HeatmapData schema:
        {
            "width": grid_width,
            "height": grid_height,
            "grid": [[float, ...], ...],  # normalized 0-1
            "video_width": video_width,
            "video_height": video_height,
        }
    """
    grid = np.zeros((grid_height, grid_width), dtype=np.float64)

    count = 0
    for det in detections:
        if class_filter and det["class_name"] != class_filter:
            continue

        bbox = det["bbox"]
        # Calculate bounding box center
        cx = (bbox["x1"] + bbox["x2"]) / 2.0
        cy = (bbox["y1"] + bbox["y2"]) / 2.0

        # Map pixel coordinates to grid cell
        gx = int((cx / video_width) * grid_width)
        gy = int((cy / video_height) * grid_height)

        # Clamp to grid bounds
        gx = max(0, min(gx, grid_width - 1))
        gy = max(0, min(gy, grid_height - 1))

        # Also add some spread based on bbox size for smoother heatmaps
        bw = (bbox["x2"] - bbox["x1"]) / video_width * grid_width
        bh = (bbox["y2"] - bbox["y1"]) / video_height * grid_height
        spread_x = max(1, int(bw / 4))
        spread_y = max(1, int(bh / 4))

        for dy in range(-spread_y, spread_y + 1):
            for dx in range(-spread_x, spread_x + 1):
                ny, nx = gy + dy, gx + dx
                if 0 <= ny < grid_height and 0 <= nx < grid_width:
                    # Gaussian-like falloff
                    dist = (dx ** 2 + dy ** 2) / max(1, spread_x ** 2 + spread_y ** 2)
                    weight = max(0, 1.0 - dist)
                    grid[ny, nx] += weight

        count += 1

    # Normalize to 0-1
    max_val = grid.max()
    if max_val > 0:
        grid = grid / max_val

    logger.info(f"Generated heatmap ({grid_width}x{grid_height}) from {count} detections")

    return {
        "width": grid_width,
        "height": grid_height,
        "grid": grid.round(4).tolist(),
        "video_width": video_width,
        "video_height": video_height,
    }
