"use client";

import { Badge } from "@/components/ui/badge";
import { VideoStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const statusConfig: Record<
  VideoStatus,
  { label: string; className: string }
> = {
  [VideoStatus.COMPLETED]: {
    label: "Completed",
    className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  },
  [VideoStatus.PROCESSING]: {
    label: "Processing",
    className: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
  },
  [VideoStatus.PENDING]: {
    label: "Pending",
    className: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  },
  [VideoStatus.FAILED]: {
    label: "Failed",
    className: "bg-rose-500/10 text-rose-400 border-rose-500/20",
  },
};

export function VideoStatusBadge({ status }: { status: VideoStatus }) {
  const config = statusConfig[status];
  return (
    <Badge variant="outline" className={cn("font-mono text-[10px]", config.className)}>
      {config.label}
    </Badge>
  );
}
