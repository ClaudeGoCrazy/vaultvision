import {
  VideoStatus,
  DetectionClass,
  EventType,
  type VideoSummary,
  type Detection,
  type Event,
  type HeatmapData,
  type AnalyticsSummary,
  type NLQueryResult,
  type WSProgressMessage,
} from "./types";

// ============================================================
// MOCK VIDEOS
// ============================================================

export const mockVideos: VideoSummary[] = [
  {
    video_id: "vid_001",
    filename: "parking_lot_cam1_2026-02-06.mp4",
    status: VideoStatus.COMPLETED,
    duration_sec: 3600,
    upload_time: "2026-02-06T08:30:00Z",
    total_detections: 1247,
    unique_persons: 34,
    unique_vehicles: 18,
    thumbnail_path: null,
  },
  {
    video_id: "vid_002",
    filename: "lobby_entrance_2026-02-06.mp4",
    status: VideoStatus.COMPLETED,
    duration_sec: 7200,
    upload_time: "2026-02-06T09:15:00Z",
    total_detections: 2891,
    unique_persons: 87,
    unique_vehicles: 0,
    thumbnail_path: null,
  },
  {
    video_id: "vid_003",
    filename: "warehouse_zone_b.mp4",
    status: VideoStatus.PROCESSING,
    duration_sec: 1800,
    upload_time: "2026-02-07T14:00:00Z",
    total_detections: 0,
    unique_persons: 0,
    unique_vehicles: 0,
    thumbnail_path: null,
  },
  {
    video_id: "vid_004",
    filename: "retail_floor_cam3.mp4",
    status: VideoStatus.COMPLETED,
    duration_sec: 5400,
    upload_time: "2026-02-05T16:45:00Z",
    total_detections: 4102,
    unique_persons: 156,
    unique_vehicles: 0,
    thumbnail_path: null,
  },
  {
    video_id: "vid_005",
    filename: "loading_dock_night.mp4",
    status: VideoStatus.PENDING,
    duration_sec: 2700,
    upload_time: "2026-02-07T15:30:00Z",
    total_detections: 0,
    unique_persons: 0,
    unique_vehicles: 0,
    thumbnail_path: null,
  },
  {
    video_id: "vid_006",
    filename: "front_gate_2026-02-04.avi",
    status: VideoStatus.COMPLETED,
    duration_sec: 4200,
    upload_time: "2026-02-04T07:00:00Z",
    total_detections: 892,
    unique_persons: 21,
    unique_vehicles: 45,
    thumbnail_path: null,
  },
  {
    video_id: "vid_007",
    filename: "stairwell_b2.mp4",
    status: VideoStatus.FAILED,
    duration_sec: 600,
    upload_time: "2026-02-07T12:00:00Z",
    total_detections: 0,
    unique_persons: 0,
    unique_vehicles: 0,
    thumbnail_path: null,
  },
  {
    video_id: "vid_008",
    filename: "intersection_main_st.mkv",
    status: VideoStatus.COMPLETED,
    duration_sec: 3000,
    upload_time: "2026-02-05T11:20:00Z",
    total_detections: 3456,
    unique_persons: 67,
    unique_vehicles: 134,
    thumbnail_path: null,
  },
];

// ============================================================
// MOCK DETECTIONS (for vid_001)
// ============================================================

export const mockDetections: Detection[] = [
  {
    detection_id: "det_001_0001",
    frame_number: 0,
    timestamp_sec: 0.0,
    class_name: DetectionClass.PERSON,
    confidence: 0.94,
    bbox: { x1: 120, y1: 200, x2: 220, y2: 480 },
    track_id: 1,
  },
  {
    detection_id: "det_001_0002",
    frame_number: 0,
    timestamp_sec: 0.0,
    class_name: DetectionClass.CAR,
    confidence: 0.97,
    bbox: { x1: 400, y1: 300, x2: 700, y2: 500 },
    track_id: 2,
  },
  {
    detection_id: "det_001_0003",
    frame_number: 2,
    timestamp_sec: 1.0,
    class_name: DetectionClass.PERSON,
    confidence: 0.91,
    bbox: { x1: 130, y1: 195, x2: 230, y2: 475 },
    track_id: 1,
  },
  {
    detection_id: "det_001_0004",
    frame_number: 4,
    timestamp_sec: 2.0,
    class_name: DetectionClass.TRUCK,
    confidence: 0.88,
    bbox: { x1: 50, y1: 280, x2: 350, y2: 520 },
    track_id: 3,
  },
  {
    detection_id: "det_001_0005",
    frame_number: 6,
    timestamp_sec: 3.0,
    class_name: DetectionClass.PERSON,
    confidence: 0.96,
    bbox: { x1: 600, y1: 150, x2: 680, y2: 420 },
    track_id: 4,
  },
  {
    detection_id: "det_001_0006",
    frame_number: 8,
    timestamp_sec: 4.0,
    class_name: DetectionClass.BACKPACK,
    confidence: 0.82,
    bbox: { x1: 210, y1: 350, x2: 270, y2: 430 },
    track_id: 5,
  },
  {
    detection_id: "det_001_0007",
    frame_number: 10,
    timestamp_sec: 5.0,
    class_name: DetectionClass.DOG,
    confidence: 0.79,
    bbox: { x1: 750, y1: 400, x2: 830, y2: 490 },
    track_id: 6,
  },
  {
    detection_id: "det_001_0008",
    frame_number: 12,
    timestamp_sec: 6.0,
    class_name: DetectionClass.PERSON,
    confidence: 0.93,
    bbox: { x1: 300, y1: 180, x2: 380, y2: 460 },
    track_id: 7,
  },
  {
    detection_id: "det_001_0009",
    frame_number: 20,
    timestamp_sec: 10.0,
    class_name: DetectionClass.CAR,
    confidence: 0.95,
    bbox: { x1: 100, y1: 320, x2: 380, y2: 510 },
    track_id: 8,
  },
  {
    detection_id: "det_001_0010",
    frame_number: 30,
    timestamp_sec: 15.0,
    class_name: DetectionClass.PERSON,
    confidence: 0.89,
    bbox: { x1: 500, y1: 200, x2: 570, y2: 470 },
    track_id: 9,
  },
];

// ============================================================
// MOCK EVENTS (for vid_001)
// ============================================================

export const mockEvents: Event[] = [
  {
    event_id: "evt_001",
    event_type: EventType.ENTRY,
    class_name: DetectionClass.PERSON,
    track_id: 1,
    start_time_sec: 0.0,
    end_time_sec: null,
    description: "Person entered frame from the left side of parking lot",
    confidence: 0.94,
    metadata: {},
  },
  {
    event_id: "evt_002",
    event_type: EventType.ENTRY,
    class_name: DetectionClass.CAR,
    track_id: 2,
    start_time_sec: 0.0,
    end_time_sec: 45.0,
    description: "Silver sedan entered parking lot from main entrance, parked in spot B12",
    confidence: 0.97,
    metadata: {},
  },
  {
    event_id: "evt_003",
    event_type: EventType.ENTRY,
    class_name: DetectionClass.TRUCK,
    track_id: 3,
    start_time_sec: 2.0,
    end_time_sec: 120.0,
    description: "White delivery truck entered from loading dock access road",
    confidence: 0.88,
    metadata: {},
  },
  {
    event_id: "evt_004",
    event_type: EventType.LOITERING,
    class_name: DetectionClass.PERSON,
    track_id: 4,
    start_time_sec: 3.0,
    end_time_sec: 180.0,
    description: "Person loitering near emergency exit for 3 minutes",
    confidence: 0.87,
    metadata: { zone: "emergency_exit_a" },
  },
  {
    event_id: "evt_005",
    event_type: EventType.EXIT,
    class_name: DetectionClass.PERSON,
    track_id: 1,
    start_time_sec: 42.0,
    end_time_sec: null,
    description: "Person exited frame through south gate",
    confidence: 0.91,
    metadata: {},
  },
  {
    event_id: "evt_006",
    event_type: EventType.ZONE_INTRUSION,
    class_name: DetectionClass.PERSON,
    track_id: 7,
    start_time_sec: 6.0,
    end_time_sec: 8.5,
    description: "Unauthorized person entered restricted loading zone",
    confidence: 0.93,
    metadata: { zone: "restricted_loading" },
  },
  {
    event_id: "evt_007",
    event_type: EventType.OBJECT_LEFT,
    class_name: DetectionClass.BACKPACK,
    track_id: 5,
    start_time_sec: 4.0,
    end_time_sec: 300.0,
    description: "Backpack left unattended near bench for 5 minutes",
    confidence: 0.82,
    metadata: { location: "bench_area" },
  },
  {
    event_id: "evt_008",
    event_type: EventType.CROWD_THRESHOLD,
    class_name: DetectionClass.PERSON,
    track_id: null,
    start_time_sec: 600.0,
    end_time_sec: 720.0,
    description: "Crowd of 12+ people gathered near main entrance exceeding threshold of 10",
    confidence: 0.95,
    metadata: { count: 12, threshold: 10 },
  },
];

// ============================================================
// MOCK HEATMAP (for vid_001)
// ============================================================

function generateHeatmapGrid(rows: number, cols: number): number[][] {
  const grid: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      // Create hotspots near entrances and walkways
      const cx1 = cols * 0.2, cy1 = rows * 0.5; // left entrance
      const cx2 = cols * 0.8, cy2 = rows * 0.3; // right area
      const cx3 = cols * 0.5, cy3 = rows * 0.7; // center walkway

      const d1 = Math.sqrt((c - cx1) ** 2 + (r - cy1) ** 2) / (cols * 0.15);
      const d2 = Math.sqrt((c - cx2) ** 2 + (r - cy2) ** 2) / (cols * 0.2);
      const d3 = Math.sqrt((c - cx3) ** 2 + (r - cy3) ** 2) / (cols * 0.25);

      const val = Math.max(
        Math.exp(-d1 * d1),
        Math.exp(-d2 * d2) * 0.7,
        Math.exp(-d3 * d3) * 0.5
      );
      row.push(Math.round(val * 100) / 100);
    }
    grid.push(row);
  }
  return grid;
}

export const mockHeatmap: HeatmapData = {
  width: 32,
  height: 24,
  grid: generateHeatmapGrid(24, 32),
  video_width: 1920,
  video_height: 1080,
};

// ============================================================
// MOCK ANALYTICS SUMMARY
// ============================================================

export const mockAnalytics: AnalyticsSummary = {
  total_videos: 8,
  total_processed: 5,
  total_processing: 1,
  total_detections: 12588,
  total_unique_persons: 365,
  total_unique_vehicles: 197,
  total_events: 1842,
  total_processing_hours: 6.5,
};

// ============================================================
// MOCK NL QUERY RESULTS
// ============================================================

export const mockQueryResults: NLQueryResult[] = [
  {
    event: mockEvents[2],
    video_id: "vid_001",
    video_filename: "parking_lot_cam1_2026-02-06.mp4",
    relevance_score: 0.95,
    thumbnail_path: null,
  },
  {
    event: mockEvents[1],
    video_id: "vid_001",
    video_filename: "parking_lot_cam1_2026-02-06.mp4",
    relevance_score: 0.82,
    thumbnail_path: null,
  },
  {
    event: {
      event_id: "evt_ext_001",
      event_type: EventType.ENTRY,
      class_name: DetectionClass.TRUCK,
      track_id: 15,
      start_time_sec: 1200.0,
      end_time_sec: 1350.0,
      description: "Red pickup truck entered from east gate at night",
      confidence: 0.91,
      metadata: {},
    },
    video_id: "vid_006",
    video_filename: "front_gate_2026-02-04.avi",
    relevance_score: 0.78,
    thumbnail_path: null,
  },
  {
    event: {
      event_id: "evt_ext_002",
      event_type: EventType.EXIT,
      class_name: DetectionClass.CAR,
      track_id: 22,
      start_time_sec: 2400.0,
      end_time_sec: null,
      description: "Dark SUV exited through main gate after 10pm",
      confidence: 0.86,
      metadata: {},
    },
    video_id: "vid_008",
    video_filename: "intersection_main_st.mkv",
    relevance_score: 0.71,
    thumbnail_path: null,
  },
];

// ============================================================
// MOCK WS PROGRESS (for simulating real-time updates)
// ============================================================

export const mockProgressMessages: WSProgressMessage[] = [
  {
    type: "progress",
    video_id: "vid_003",
    status: VideoStatus.PROCESSING,
    progress_percent: 35,
    current_step: "Running YOLOv8 object detection",
    message: "Processing frame 252/720",
  },
  {
    type: "progress",
    video_id: "vid_003",
    status: VideoStatus.PROCESSING,
    progress_percent: 55,
    current_step: "ByteTrack multi-object tracking",
    message: "Tracking objects across frames",
  },
  {
    type: "progress",
    video_id: "vid_003",
    status: VideoStatus.PROCESSING,
    progress_percent: 78,
    current_step: "Generating heatmap",
    message: "Computing spatial density matrix",
  },
];
