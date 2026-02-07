"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import {
  Video,
  Upload,
  Grid3X3,
  List,
  Search,
  Trash2,
  Eye,
  Users,
  Car,
  Filter,
  Loader2,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { VideoStatusBadge } from "@/components/video-status-badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { VideoStatus } from "@/lib/types";
import type { VideoSummary } from "@/lib/types";
import { formatDuration, formatDate, formatNumber } from "@/lib/format";

export default function VideosPage() {
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: videos, loading, refetch } = useApi<VideoSummary[]>(
    () => api.listVideos(),
    []
  );

  // Auto-refresh every 5 seconds to catch status changes
  useEffect(() => {
    const interval = setInterval(refetch, 5000);
    return () => clearInterval(interval);
  }, [refetch]);

  const allVideos = videos || [];

  const filtered = allVideos.filter((v) => {
    const matchesSearch = v.filename
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesStatus =
      statusFilter === "all" || v.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const completedCount = allVideos.filter(
    (v) => v.status === VideoStatus.COMPLETED
  ).length;

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      await api.uploadVideo(file);
      refetch();
    } catch (err) {
      console.error("Upload failed:", err);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleDelete = async (videoId: string) => {
    try {
      await api.deleteVideo(videoId);
      refetch();
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Video Library</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {allVideos.length} videos &middot; {completedCount} processed
            {loading && <Loader2 className="ml-2 inline h-3 w-3 animate-spin" />}
          </p>
        </div>
        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/mp4,video/avi,video/x-matroska,video/quicktime"
            className="hidden"
            onChange={handleUpload}
          />
          <Button
            className="gap-2"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            {uploading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Upload className="h-4 w-4" />
            )}
            {uploading ? "Uploading..." : "Upload Video"}
          </Button>
        </div>
      </div>

      {/* Toolbar */}
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search videos..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 bg-card/50 border-border/50"
          />
        </div>
        <div className="flex items-center gap-2">
          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-[140px] bg-card/50 border-border/50">
              <Filter className="mr-2 h-3.5 w-3.5 text-muted-foreground" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value={VideoStatus.COMPLETED}>Completed</SelectItem>
              <SelectItem value={VideoStatus.PROCESSING}>Processing</SelectItem>
              <SelectItem value={VideoStatus.PENDING}>Pending</SelectItem>
              <SelectItem value={VideoStatus.FAILED}>Failed</SelectItem>
            </SelectContent>
          </Select>
          <div className="flex rounded-md border border-border/50">
            <Button
              variant={viewMode === "grid" ? "secondary" : "ghost"}
              size="icon"
              className="h-9 w-9 rounded-r-none"
              onClick={() => setViewMode("grid")}
            >
              <Grid3X3 className="h-4 w-4" />
            </Button>
            <Button
              variant={viewMode === "list" ? "secondary" : "ghost"}
              size="icon"
              className="h-9 w-9 rounded-l-none"
              onClick={() => setViewMode("list")}
            >
              <List className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Grid View */}
      {viewMode === "grid" && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filtered.map((video) => (
            <Link key={video.video_id} href={`/videos/${video.video_id}`}>
              <Card className="group border-border/50 bg-card/50 transition-colors hover:border-primary/30 hover:bg-card/80">
                <div className="relative aspect-video w-full overflow-hidden rounded-t-lg bg-background/50">
                  <div className="flex h-full items-center justify-center">
                    <Video className="h-10 w-10 text-muted-foreground/30" />
                  </div>
                  <div className="absolute bottom-2 right-2">
                    <VideoStatusBadge status={video.status} />
                  </div>
                  {video.duration_sec && (
                    <span className="absolute bottom-2 left-2 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white">
                      {formatDuration(video.duration_sec)}
                    </span>
                  )}
                </div>
                <CardContent className="p-3">
                  <p className="truncate font-mono text-xs font-medium">
                    {video.filename}
                  </p>
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    {formatDate(video.upload_time)}
                  </p>
                  {video.status === VideoStatus.COMPLETED && (
                    <div className="mt-2 flex items-center gap-3 text-[10px] text-muted-foreground">
                      <span className="flex items-center gap-1">
                        <Eye className="h-3 w-3" />
                        {formatNumber(video.total_detections)}
                      </span>
                      <span className="flex items-center gap-1">
                        <Users className="h-3 w-3" />
                        {video.unique_persons}
                      </span>
                      <span className="flex items-center gap-1">
                        <Car className="h-3 w-3" />
                        {video.unique_vehicles}
                      </span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}

      {/* List View */}
      {viewMode === "list" && (
        <Card className="border-border/50 bg-card/50">
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
                    <th className="px-4 py-3 text-left">Filename</th>
                    <th className="px-4 py-3 text-left">Status</th>
                    <th className="px-4 py-3 text-right">Duration</th>
                    <th className="px-4 py-3 text-right">Detections</th>
                    <th className="px-4 py-3 text-right">Persons</th>
                    <th className="px-4 py-3 text-right">Vehicles</th>
                    <th className="px-4 py-3 text-left">Uploaded</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((video) => (
                    <tr
                      key={video.video_id}
                      className="border-b border-border/30 transition-colors hover:bg-accent/30"
                    >
                      <td className="px-4 py-3">
                        <Link
                          href={`/videos/${video.video_id}`}
                          className="flex items-center gap-2 font-mono text-xs hover:text-primary"
                        >
                          <Video className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                          <span className="truncate max-w-[200px]">
                            {video.filename}
                          </span>
                        </Link>
                      </td>
                      <td className="px-4 py-3">
                        <VideoStatusBadge status={video.status} />
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground">
                        {formatDuration(video.duration_sec)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground">
                        {video.status === VideoStatus.COMPLETED
                          ? formatNumber(video.total_detections)
                          : "--"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground">
                        {video.status === VideoStatus.COMPLETED
                          ? video.unique_persons
                          : "--"}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground">
                        {video.status === VideoStatus.COMPLETED
                          ? video.unique_vehicles
                          : "--"}
                      </td>
                      <td className="px-4 py-3 text-xs text-muted-foreground">
                        {formatDate(video.upload_time)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-7 w-7 text-muted-foreground hover:text-destructive"
                          onClick={(e) => {
                            e.preventDefault();
                            handleDelete(video.video_id);
                          }}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {filtered.length === 0 && !loading && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <Video className="mb-4 h-12 w-12 text-muted-foreground/30" />
          <p className="text-sm text-muted-foreground">No videos found</p>
          <p className="mt-1 text-xs text-muted-foreground/60">
            Upload a video or adjust your search
          </p>
        </div>
      )}
    </div>
  );
}
