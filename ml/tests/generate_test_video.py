"""
Generate a synthetic test video with moving rectangles simulating people/objects.
This creates a video that YOLOv8 can process without needing a real surveillance video.
"""
import cv2
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "uploads"


def generate_test_video(
    output_path: str = None,
    duration_sec: int = 10,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Generate a test video with moving colored shapes."""
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / "test_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    # Define some "objects" that move across the frame
    objects = [
        {"x": 100, "y": 200, "w": 60, "h": 120, "dx": 3, "dy": 1, "color": (0, 255, 0), "label": "person1"},
        {"x": 800, "y": 300, "w": 60, "h": 120, "dx": -2, "dy": 1, "color": (0, 200, 0), "label": "person2"},
        {"x": 400, "y": 500, "w": 150, "h": 80, "dx": 4, "dy": 0, "color": (255, 0, 0), "label": "car"},
        {"x": 600, "y": 100, "w": 50, "h": 100, "dx": 1, "dy": 2, "color": (0, 180, 0), "label": "person3"},
    ]

    for frame_idx in range(total_frames):
        # Background - outdoor scene-like gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky
        frame[:height // 2, :] = [200, 180, 140]  # light blue-ish
        # Ground
        frame[height // 2:, :] = [80, 120, 80]  # green-ish

        # Add some static "buildings"
        cv2.rectangle(frame, (50, 100), (200, height // 2), (150, 150, 150), -1)
        cv2.rectangle(frame, (900, 150), (1050, height // 2), (160, 160, 160), -1)
        cv2.rectangle(frame, (1100, 80), (1200, height // 2), (140, 140, 140), -1)

        # Draw and move objects
        for obj in objects:
            x, y = int(obj["x"]), int(obj["y"])
            w, h = obj["w"], obj["h"]

            # Draw the shape
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], -1)

            # Add a "head" circle for person-like shapes
            if "person" in obj["label"]:
                cv2.circle(frame, (x + w // 2, y - 15), 15, obj["color"], -1)

            # Move
            obj["x"] += obj["dx"]
            obj["y"] += obj["dy"]

            # Bounce off edges
            if obj["x"] <= 0 or obj["x"] + w >= width:
                obj["dx"] *= -1
            if obj["y"] <= 0 or obj["y"] + h >= height:
                obj["dy"] *= -1
            obj["x"] = max(0, min(obj["x"], width - w))
            obj["y"] = max(0, min(obj["y"], height - h))

        # Add timestamp text
        timestamp = frame_idx / fps
        cv2.putText(
            frame,
            f"Time: {timestamp:.1f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

    writer.release()
    print(f"Generated test video: {output_path} ({duration_sec}s, {fps}fps, {width}x{height})")
    return output_path


if __name__ == "__main__":
    generate_test_video()
