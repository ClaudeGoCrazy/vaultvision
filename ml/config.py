"""
VaultVision ML Pipeline Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ML_ROOT = Path(__file__).parent
PROJECT_ROOT = ML_ROOT.parent
DATA_DIR = ML_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = ML_ROOT / "models"

# Ensure dirs exist
for d in [UPLOADS_DIR, FRAMES_DIR, OUTPUTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Frame Extraction
DEFAULT_FPS = 2.0  # Frames per second to extract
MAX_FRAMES_PER_VIDEO = 10000  # Safety cap for very long videos

# YOLOv8 - Model selection: n(ano), s(mall), m(edium), l(arge), x(tra-large)
# Larger models = better accuracy, slower speed
YOLO_MODEL = os.getenv("VAULTVISION_YOLO_MODEL", "yolov8s.pt")  # small for good balance
YOLO_CONFIDENCE_THRESHOLD = float(os.getenv("VAULTVISION_YOLO_CONF", "0.30"))
YOLO_IOU_THRESHOLD = float(os.getenv("VAULTVISION_IOU_THRESH", "0.50"))
YOLO_MAX_DETECTIONS = int(os.getenv("VAULTVISION_MAX_DET", "100"))
YOLO_IMGSZ = int(os.getenv("VAULTVISION_IMGSZ", "640"))

# Device auto-detection
_device_env = os.getenv("VAULTVISION_DEVICE", "auto")
if _device_env == "auto":
    try:
        import torch
        YOLO_DEVICE = "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        YOLO_DEVICE = "cpu"
else:
    YOLO_DEVICE = _device_env

# ByteTrack
TRACK_HIGH_THRESH = 0.5
TRACK_LOW_THRESH = 0.1
TRACK_MATCH_THRESH = 0.8
TRACK_BUFFER = 30  # frames to keep lost tracks

# Heatmap
HEATMAP_GRID_WIDTH = 64
HEATMAP_GRID_HEIGHT = 48

# ChromaDB
CHROMA_PERSIST_DIR = str(DATA_DIR / "chromadb")
CHROMA_COLLECTION_NAME = "vaultvision_events"

# Anthropic / Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# Redis (for Celery)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Anomaly Detection
LOITERING_THRESHOLD_SEC = float(os.getenv("VAULTVISION_LOITER_THRESH", "30.0"))
CROWD_THRESHOLD = int(os.getenv("VAULTVISION_CROWD_THRESH", "5"))
ZONE_INTRUSION_ENABLED = os.getenv("VAULTVISION_ZONES_ENABLED", "false").lower() == "true"

# Object-left-behind detection
OBJECT_LEFT_THRESHOLD_SEC = float(os.getenv("VAULTVISION_OBJ_LEFT_THRESH", "60.0"))
OBJECT_LEFT_CLASSES = {"backpack", "suitcase", "handbag"}

# Loitering spatial threshold (fraction of frame diagonal)
LOITERING_SPATIAL_THRESH = 0.15  # object must stay within 15% of frame diagonal
