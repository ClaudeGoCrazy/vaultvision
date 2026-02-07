// VaultVision Frontend Types
// Mirrors ~/vaultvision/shared/schemas.py exactly

export enum VideoStatus {
  PENDING = "pending",
  PROCESSING = "processing",
  COMPLETED = "completed",
  FAILED = "failed",
}

export enum DetectionClass {
  PERSON = "person",
  VEHICLE = "vehicle",
  BICYCLE = "bicycle",
  MOTORCYCLE = "motorcycle",
  BUS = "bus",
  TRUCK = "truck",
  CAR = "car",
  DOG = "dog",
  CAT = "cat",
  BACKPACK = "backpack",
  HANDBAG = "handbag",
  SUITCASE = "suitcase",
  CELL_PHONE = "cell_phone",
  LICENSE_PLATE = "license_plate",
  OTHER = "other",
}

export enum EventType {
  ENTRY = "entry",
  EXIT = "exit",
  LOITERING = "loitering",
  ZONE_INTRUSION = "zone_intrusion",
  CROWD_THRESHOLD = "crowd_threshold",
  OBJECT_LEFT = "object_left",
  ANOMALY = "anomaly",
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  detection_id: string;
  frame_number: number;
  timestamp_sec: number;
  class_name: DetectionClass;
  confidence: number;
  bbox: BoundingBox;
  track_id: number | null;
}

export interface Event {
  event_id: string;
  event_type: EventType;
  class_name: DetectionClass;
  track_id: number | null;
  start_time_sec: number;
  end_time_sec: number | null;
  description: string;
  confidence: number;
  metadata: Record<string, unknown>;
}

export interface HeatmapData {
  width: number;
  height: number;
  grid: number[][];
  video_width: number;
  video_height: number;
}

export interface PipelineResult {
  video_id: string;
  total_frames: number;
  fps_processed: number;
  processing_time_sec: number;
  detections: Detection[];
  events: Event[];
  heatmap: HeatmapData;
  unique_person_count: number;
  unique_vehicle_count: number;
  object_class_counts: Record<string, number>;
}

export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  status: VideoStatus;
  message: string;
}

export interface VideoStatusResponse {
  video_id: string;
  status: VideoStatus;
  progress_percent: number;
  current_step: string | null;
  estimated_remaining_sec: number | null;
}

export interface VideoSummary {
  video_id: string;
  filename: string;
  status: VideoStatus;
  duration_sec: number | null;
  upload_time: string;
  total_detections: number;
  unique_persons: number;
  unique_vehicles: number;
  thumbnail_path: string | null;
}

export interface NLQueryRequest {
  query: string;
  video_id?: string;
  limit: number;
}

export interface NLQueryResult {
  event: Event;
  video_id: string;
  video_filename: string;
  relevance_score: number;
  thumbnail_path: string | null;
}

export interface NLQueryResponse {
  query: string;
  results: NLQueryResult[];
  total_results: number;
  processing_time_ms: number;
}

export interface AnalyticsSummary {
  total_videos: number;
  total_processed: number;
  total_processing: number;
  total_detections: number;
  total_unique_persons: number;
  total_unique_vehicles: number;
  total_events: number;
  total_processing_hours: number;
}

export interface WSProgressMessage {
  type: "progress";
  video_id: string;
  status: VideoStatus;
  progress_percent: number;
  current_step: string;
  message: string | null;
}
