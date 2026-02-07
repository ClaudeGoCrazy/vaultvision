"use client";

import { useEffect, useMemo } from "react";
import {
  Video,
  Users,
  Car,
  Eye,
  AlertTriangle,
  Clock,
  Upload,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/stat-card";
import { VideoStatusBadge } from "@/components/video-status-badge";
import { ProcessingQueue } from "@/components/processing-queue";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";
import { VideoStatus } from "@/lib/types";
import type { VideoSummary, AnalyticsSummary, Event } from "@/lib/types";
import { formatDuration, formatDate, formatNumber } from "@/lib/format";
import Link from "next/link";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function DashboardPage() {
  const { data: videos, loading: videosLoading, refetch: refetchVideos } = useApi<VideoSummary[]>(
    () => api.listVideos(),
    []
  );
  const { data: analytics, loading: analyticsLoading, refetch: refetchAnalytics } = useApi<AnalyticsSummary>(
    () => api.getAnalyticsSummary(),
    []
  );

  // Auto-refresh every 10 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      refetchVideos();
      refetchAnalytics();
    }, 10000);
    return () => clearInterval(interval);
  }, [refetchVideos, refetchAnalytics]);

  const recentVideos = useMemo(() => {
    if (!videos) return [];
    return [...videos]
      .sort(
        (a, b) =>
          new Date(b.upload_time).getTime() - new Date(a.upload_time).getTime()
      )
      .slice(0, 5);
  }, [videos]);

  // Aggregate detection class counts from all completed video detections
  // For now we use the analytics summary; a per-class breakdown would require a new endpoint
  const stats = analytics || {
    total_videos: 0,
    total_processed: 0,
    total_processing: 0,
    total_detections: 0,
    total_unique_persons: 0,
    total_unique_vehicles: 0,
    total_events: 0,
    total_processing_hours: 0,
  };

  // Build a simple chart from video-level data
  const chartData = useMemo(() => {
    if (!videos) return [];
    const completed = videos.filter((v) => v.status === VideoStatus.COMPLETED);
    return completed.map((v) => ({
      name: v.filename.length > 15 ? v.filename.slice(0, 12) + "..." : v.filename,
      detections: v.total_detections,
      persons: v.unique_persons,
      vehicles: v.unique_vehicles,
    }));
  }, [videos]);

  const isLoading = videosLoading || analyticsLoading;

  return (
    <div className="p-6 lg:p-8">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            VaultVision command center overview
            {isLoading && (
              <Loader2 className="ml-2 inline h-3 w-3 animate-spin" />
            )}
          </p>
        </div>
        <Link href="/videos">
          <Button className="gap-2">
            <Upload className="h-4 w-4" />
            Upload Video
          </Button>
        </Link>
      </div>

      {/* Stats Grid */}
      <div className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Total Videos"
          value={formatNumber(stats.total_videos)}
          icon={Video}
          trend={`${stats.total_processed} processed`}
          accent="cyan"
        />
        <StatCard
          label="Total Detections"
          value={formatNumber(stats.total_detections)}
          icon={Eye}
          trend={`${formatNumber(stats.total_events)} events`}
          accent="emerald"
        />
        <StatCard
          label="Unique Persons"
          value={formatNumber(stats.total_unique_persons)}
          icon={Users}
          trend="Across all videos"
          accent="amber"
        />
        <StatCard
          label="Unique Vehicles"
          value={formatNumber(stats.total_unique_vehicles)}
          icon={Car}
          trend={`${stats.total_processing_hours}h processed`}
          accent="rose"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Recent Videos - spans 2 cols */}
        <div className="lg:col-span-2">
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Recent Videos
                </CardTitle>
                <Link href="/videos">
                  <Button variant="ghost" size="sm" className="text-xs text-muted-foreground">
                    View All
                  </Button>
                </Link>
              </div>
            </CardHeader>
            <CardContent>
              {recentVideos.length === 0 && !videosLoading && (
                <p className="py-8 text-center text-sm text-muted-foreground/60">
                  No videos uploaded yet. Upload a video to get started.
                </p>
              )}
              {recentVideos.length > 0 && (
                <div className="space-y-1">
                  <div className="grid grid-cols-[1fr_100px_80px_80px_80px] gap-2 border-b border-border/50 pb-2 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
                    <span>Filename</span>
                    <span>Status</span>
                    <span className="text-right">Duration</span>
                    <span className="text-right">Persons</span>
                    <span className="text-right">Vehicles</span>
                  </div>
                  {recentVideos.map((video) => (
                    <Link
                      key={video.video_id}
                      href={`/videos/${video.video_id}`}
                      className="grid grid-cols-[1fr_100px_80px_80px_80px] gap-2 rounded-md px-1 py-2.5 transition-colors hover:bg-accent/50"
                    >
                      <div className="flex items-center gap-2 overflow-hidden">
                        <Video className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                        <span className="truncate font-mono text-xs">
                          {video.filename}
                        </span>
                      </div>
                      <div>
                        <VideoStatusBadge status={video.status} />
                      </div>
                      <span className="text-right font-mono text-xs text-muted-foreground">
                        {formatDuration(video.duration_sec)}
                      </span>
                      <span className="text-right font-mono text-xs text-muted-foreground">
                        {video.status === VideoStatus.COMPLETED
                          ? video.unique_persons
                          : "--"}
                      </span>
                      <span className="text-right font-mono text-xs text-muted-foreground">
                        {video.status === VideoStatus.COMPLETED
                          ? video.unique_vehicles
                          : "--"}
                      </span>
                    </Link>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Detection Chart */}
          {chartData.length > 0 && (
            <Card className="mt-6 border-border/50 bg-card/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Detections per Video
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[240px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barCategoryGap="20%">
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="oklch(0.25 0.01 260)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="name"
                        tick={{ fontSize: 11, fill: "oklch(0.60 0.02 260)" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: "oklch(0.60 0.02 260)" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "oklch(0.16 0.01 260)",
                          border: "1px solid oklch(0.25 0.01 260)",
                          borderRadius: "8px",
                          fontSize: "12px",
                          fontFamily: "var(--font-geist-mono)",
                        }}
                        itemStyle={{ color: "oklch(0.93 0.01 260)" }}
                        labelStyle={{ color: "oklch(0.60 0.02 260)" }}
                      />
                      <Bar
                        dataKey="detections"
                        fill="oklch(0.72 0.15 195)"
                        radius={[4, 4, 0, 0]}
                        name="Detections"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Processing Queue */}
          <ProcessingQueue videos={videos || []} />

          {/* Quick Stats */}
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Processing Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  Total Hours Processed
                </span>
                <span className="font-mono text-sm font-medium text-foreground">
                  {stats.total_processing_hours}h
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  Videos Completed
                </span>
                <span className="font-mono text-sm font-medium text-emerald-400">
                  {stats.total_processed}/{stats.total_videos}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  Active Processing
                </span>
                <span className="font-mono text-sm font-medium text-cyan-400">
                  {stats.total_processing}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  Total Events
                </span>
                <span className="font-mono text-sm font-medium text-foreground">
                  {formatNumber(stats.total_events)}
                </span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
