"use client";

import { useState, useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Eye,
  Users,
  Car,
  Clock,
  Layers,
  MapPin,
  AlertTriangle,
  Download,
  Flame,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VideoStatusBadge } from "@/components/video-status-badge";
import { HeatmapCanvas } from "@/components/heatmap-canvas";
import { DetectionTimeline } from "@/components/detection-timeline";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { DetectionClass, EventType, VideoStatus } from "@/lib/types";
import type { VideoSummary, Detection, Event, HeatmapData } from "@/lib/types";
import { formatTimestamp, formatDuration, formatDate } from "@/lib/format";
import { cn } from "@/lib/utils";

const eventTypeConfig: Record<EventType, { icon: typeof Eye; color: string }> = {
  [EventType.ENTRY]: { icon: ArrowLeft, color: "text-emerald-400" },
  [EventType.EXIT]: { icon: ArrowLeft, color: "text-blue-400" },
  [EventType.LOITERING]: { icon: Clock, color: "text-amber-400" },
  [EventType.ZONE_INTRUSION]: { icon: AlertTriangle, color: "text-rose-400" },
  [EventType.CROWD_THRESHOLD]: { icon: Users, color: "text-orange-400" },
  [EventType.OBJECT_LEFT]: { icon: MapPin, color: "text-purple-400" },
  [EventType.ANOMALY]: { icon: AlertTriangle, color: "text-red-400" },
};

export default function VideoAnalysisPage() {
  const params = useParams();
  const videoId = params.id as string;

  // Fetch all data from API
  const { data: videos } = useApi<VideoSummary[]>(() => api.listVideos(), []);
  const { data: detections, loading: detectionsLoading } = useApi<Detection[]>(
    () => api.getVideoDetections(videoId),
    [videoId]
  );
  const { data: events, loading: eventsLoading } = useApi<Event[]>(
    () => api.getVideoEvents(videoId),
    [videoId]
  );
  const { data: heatmap } = useApi<HeatmapData>(
    () => api.getVideoHeatmap(videoId),
    [videoId]
  );

  const video = videos?.find((v) => v.video_id === videoId);
  const videoDetections = detections || [];
  const videoEvents = events || [];

  const [currentTime, setCurrentTime] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showBboxes, setShowBboxes] = useState(true);
  const [selectedTab, setSelectedTab] = useState("detections");

  const currentDetections = useMemo(() => {
    return videoDetections.filter(
      (d) => Math.abs(d.timestamp_sec - currentTime) < 0.6
    );
  }, [videoDetections, currentTime]);

  const classCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    const seenTracks = new Set<number>();
    for (const d of videoDetections) {
      if (d.track_id && seenTracks.has(d.track_id)) continue;
      if (d.track_id) seenTracks.add(d.track_id);
      counts[d.class_name] = (counts[d.class_name] || 0) + 1;
    }
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [videoDetections]);

  const totalDuration = video?.duration_sec || 60;
  const streamUrl = api.getVideoStreamUrl(videoId);

  const bboxColor = (cls: DetectionClass) => {
    const colors: Partial<Record<DetectionClass, string>> = {
      [DetectionClass.PERSON]: "#22d3ee",
      [DetectionClass.CAR]: "#60a5fa",
      [DetectionClass.TRUCK]: "#818cf8",
      [DetectionClass.DOG]: "#fbbf24",
      [DetectionClass.BACKPACK]: "#f87171",
    };
    return colors[cls] || "#9ca3af";
  };

  if (!video) {
    return (
      <div className="flex h-full items-center justify-center p-8">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-6 flex items-center gap-4">
        <Link href="/videos">
          <Button variant="ghost" size="icon" className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </Link>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3">
            <h1 className="truncate text-xl font-bold tracking-tight font-mono">
              {video.filename}
            </h1>
            <VideoStatusBadge status={video.status} />
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Uploaded {formatDate(video.upload_time)} &middot;{" "}
            Duration {formatDuration(video.duration_sec)}
          </p>
        </div>
        <Button variant="outline" size="sm" className="gap-1.5">
          <Download className="h-3.5 w-3.5" />
          Export
        </Button>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_380px]">
        {/* Video Player Area */}
        <div className="space-y-4">
          <Card className="border-border/50 bg-card/50 overflow-hidden">
            <div className="relative aspect-video w-full bg-black">
              {/* Real video player */}
              {video.status === VideoStatus.COMPLETED ? (
                <video
                  className="h-full w-full"
                  controls
                  src={streamUrl}
                  onTimeUpdate={(e) =>
                    setCurrentTime((e.target as HTMLVideoElement).currentTime)
                  }
                />
              ) : (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    {video.status === VideoStatus.PROCESSING ? (
                      <>
                        <Loader2 className="mx-auto mb-2 h-10 w-10 animate-spin text-primary" />
                        <p className="font-mono text-xs text-muted-foreground">
                          Processing video...
                        </p>
                      </>
                    ) : (
                      <>
                        <div className="mx-auto mb-2 h-16 w-16 rounded-full border-2 border-muted-foreground/20 flex items-center justify-center">
                          <div className="h-0 w-0 ml-1 border-y-[10px] border-y-transparent border-l-[16px] border-l-muted-foreground/40" />
                        </div>
                        <p className="font-mono text-xs text-muted-foreground/40">
                          {video.status === VideoStatus.PENDING
                            ? "Pending processing..."
                            : "Video unavailable"}
                        </p>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Bounding Box Overlays */}
              {showBboxes &&
                currentDetections.map((d) => {
                  const scaleX = 100 / 1920;
                  const scaleY = 100 / 1080;
                  return (
                    <div
                      key={d.detection_id}
                      className="absolute border-2 pointer-events-none"
                      style={{
                        left: `${d.bbox.x1 * scaleX}%`,
                        top: `${d.bbox.y1 * scaleY}%`,
                        width: `${(d.bbox.x2 - d.bbox.x1) * scaleX}%`,
                        height: `${(d.bbox.y2 - d.bbox.y1) * scaleY}%`,
                        borderColor: bboxColor(d.class_name),
                      }}
                    >
                      <span
                        className="absolute -top-5 left-0 rounded px-1 py-0.5 font-mono text-[9px] text-white"
                        style={{ backgroundColor: bboxColor(d.class_name) }}
                      >
                        {d.class_name} {(d.confidence * 100).toFixed(0)}%
                        {d.track_id != null && ` #${d.track_id}`}
                      </span>
                    </div>
                  );
                })}

              {/* Heatmap overlay */}
              {showHeatmap && heatmap && (
                <HeatmapCanvas
                  data={heatmap}
                  width={960}
                  height={540}
                  opacity={0.5}
                />
              )}
            </div>
          </Card>

          {/* Overlay Controls */}
          <div className="flex items-center gap-3">
            <Button
              variant={showBboxes ? "default" : "outline"}
              size="sm"
              className="gap-1.5 text-xs"
              onClick={() => setShowBboxes(!showBboxes)}
            >
              <Layers className="h-3 w-3" />
              Bounding Boxes
            </Button>
            <Button
              variant={showHeatmap ? "default" : "outline"}
              size="sm"
              className="gap-1.5 text-xs"
              onClick={() => setShowHeatmap(!showHeatmap)}
            >
              <Flame className="h-3 w-3" />
              Heatmap
            </Button>
          </div>

          {/* Detection Timeline */}
          {videoDetections.length > 0 && (
            <Card className="border-border/50 bg-card/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs font-medium text-muted-foreground">
                  Detection Timeline
                </CardTitle>
              </CardHeader>
              <CardContent>
                <DetectionTimeline
                  detections={videoDetections}
                  totalDuration={totalDuration}
                  currentTime={currentTime}
                  onSeek={setCurrentTime}
                />
              </CardContent>
            </Card>
          )}

          {/* Stats Row */}
          <div className="grid grid-cols-4 gap-3">
            <Card className="border-border/50 bg-card/50">
              <CardContent className="p-3 text-center">
                <Eye className="mx-auto h-4 w-4 text-cyan-400" />
                <p className="mt-1 font-mono text-lg font-bold">
                  {video.total_detections.toLocaleString()}
                </p>
                <p className="text-[10px] text-muted-foreground">Detections</p>
              </CardContent>
            </Card>
            <Card className="border-border/50 bg-card/50">
              <CardContent className="p-3 text-center">
                <Users className="mx-auto h-4 w-4 text-emerald-400" />
                <p className="mt-1 font-mono text-lg font-bold">
                  {video.unique_persons}
                </p>
                <p className="text-[10px] text-muted-foreground">Persons</p>
              </CardContent>
            </Card>
            <Card className="border-border/50 bg-card/50">
              <CardContent className="p-3 text-center">
                <Car className="mx-auto h-4 w-4 text-blue-400" />
                <p className="mt-1 font-mono text-lg font-bold">
                  {video.unique_vehicles}
                </p>
                <p className="text-[10px] text-muted-foreground">Vehicles</p>
              </CardContent>
            </Card>
            <Card className="border-border/50 bg-card/50">
              <CardContent className="p-3 text-center">
                <AlertTriangle className="mx-auto h-4 w-4 text-amber-400" />
                <p className="mt-1 font-mono text-lg font-bold">
                  {videoEvents.length}
                </p>
                <p className="text-[10px] text-muted-foreground">Events</p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Right Sidebar */}
        <div>
          <Tabs value={selectedTab} onValueChange={setSelectedTab}>
            <TabsList className="w-full bg-card/50">
              <TabsTrigger value="detections" className="flex-1 text-xs">
                Detections ({videoDetections.length})
              </TabsTrigger>
              <TabsTrigger value="events" className="flex-1 text-xs">
                Events ({videoEvents.length})
              </TabsTrigger>
              <TabsTrigger value="objects" className="flex-1 text-xs">
                Objects
              </TabsTrigger>
            </TabsList>

            <TabsContent value="detections" className="mt-3">
              <ScrollArea className="h-[600px]">
                {detectionsLoading && (
                  <div className="flex justify-center py-8">
                    <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  </div>
                )}
                {!detectionsLoading && videoDetections.length === 0 && (
                  <p className="py-8 text-center text-sm text-muted-foreground/60">
                    No detections yet
                  </p>
                )}
                <div className="space-y-1 pr-3">
                  {videoDetections.map((d) => (
                    <button
                      key={d.detection_id}
                      className={cn(
                        "w-full rounded-md border border-transparent p-2.5 text-left transition-colors hover:bg-accent/50",
                        Math.abs(d.timestamp_sec - currentTime) < 0.6 &&
                          "border-primary/30 bg-primary/5"
                      )}
                      onClick={() => setCurrentTime(d.timestamp_sec)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="h-2 w-2 rounded-full"
                            style={{ backgroundColor: bboxColor(d.class_name) }}
                          />
                          <span className="text-xs font-medium">
                            {d.class_name}
                          </span>
                          {d.track_id != null && (
                            <span className="font-mono text-[10px] text-muted-foreground">
                              #{d.track_id}
                            </span>
                          )}
                        </div>
                        <span className="font-mono text-[10px] text-muted-foreground">
                          {formatTimestamp(d.timestamp_sec)}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
                        <span>
                          Confidence: {(d.confidence * 100).toFixed(0)}%
                        </span>
                        <span>&middot;</span>
                        <span>Frame {d.frame_number}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="events" className="mt-3">
              <ScrollArea className="h-[600px]">
                {eventsLoading && (
                  <div className="flex justify-center py-8">
                    <Loader2 className="h-5 w-5 animate-spin text-primary" />
                  </div>
                )}
                {!eventsLoading && videoEvents.length === 0 && (
                  <p className="py-8 text-center text-sm text-muted-foreground/60">
                    No events detected
                  </p>
                )}
                <div className="space-y-2 pr-3">
                  {videoEvents.map((evt) => {
                    const config = eventTypeConfig[evt.event_type] || eventTypeConfig[EventType.ANOMALY];
                    const Icon = config.icon;
                    return (
                      <button
                        key={evt.event_id}
                        className="w-full rounded-md border border-border/50 bg-card/30 p-3 text-left transition-colors hover:bg-accent/50"
                        onClick={() => setCurrentTime(evt.start_time_sec)}
                      >
                        <div className="flex items-start gap-2">
                          <Icon
                            className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", config.color)}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <Badge
                                variant="outline"
                                className="font-mono text-[9px]"
                              >
                                {evt.event_type}
                              </Badge>
                              <Badge
                                variant="outline"
                                className="font-mono text-[9px]"
                              >
                                {evt.class_name}
                              </Badge>
                            </div>
                            <p className="mt-1 text-xs text-foreground">
                              {evt.description}
                            </p>
                            <p className="mt-1 font-mono text-[10px] text-muted-foreground">
                              {formatTimestamp(evt.start_time_sec)}
                              {evt.end_time_sec &&
                                ` — ${formatTimestamp(evt.end_time_sec)}`}
                              &nbsp;&middot;&nbsp;
                              {(evt.confidence * 100).toFixed(0)}% confidence
                            </p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="objects" className="mt-3">
              {classCounts.length === 0 && (
                <p className="py-8 text-center text-sm text-muted-foreground/60">
                  No objects detected
                </p>
              )}
              <div className="space-y-2">
                {classCounts.map(([cls, count]) => (
                  <div
                    key={cls}
                    className="flex items-center justify-between rounded-md border border-border/50 bg-card/30 p-3"
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className="h-3 w-3 rounded"
                        style={{
                          backgroundColor: bboxColor(cls as DetectionClass),
                        }}
                      />
                      <span className="text-sm font-medium capitalize">
                        {cls}
                      </span>
                    </div>
                    <span className="font-mono text-sm font-bold text-foreground">
                      {count}
                    </span>
                  </div>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
