"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { VideoStatusBadge } from "@/components/video-status-badge";
import { VideoStatus } from "@/lib/types";
import type { VideoSummary, WSProgressMessage } from "@/lib/types";
import { Loader2 } from "lucide-react";

interface ProcessingQueueProps {
  videos: VideoSummary[];
  progress?: WSProgressMessage[];
}

export function ProcessingQueue({ videos, progress = [] }: ProcessingQueueProps) {
  const activeVideos = videos.filter(
    (v) => v.status === VideoStatus.PROCESSING || v.status === VideoStatus.PENDING
  );

  if (activeVideos.length === 0) {
    return (
      <Card className="border-border/50 bg-card/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Processing Queue
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground/60">No active jobs</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin text-cyan-400" />
          Processing Queue
          <span className="font-mono text-xs text-cyan-400">
            ({activeVideos.length})
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {activeVideos.map((video) => {
          const msg = progress.find((p) => p.video_id === video.video_id);
          return (
            <div key={video.video_id} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="truncate font-mono text-xs text-foreground">
                  {video.filename}
                </span>
                <VideoStatusBadge status={video.status} />
              </div>
              {msg && (
                <>
                  <Progress value={msg.progress_percent} className="h-1.5" />
                  <p className="font-mono text-[10px] text-muted-foreground">
                    {msg.current_step} — {msg.progress_percent}%
                  </p>
                </>
              )}
              {!msg && video.status === VideoStatus.PENDING && (
                <p className="font-mono text-[10px] text-muted-foreground">
                  Queued — waiting for processing slot
                </p>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
