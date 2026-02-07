"use client";

import { useMemo } from "react";
import type { Detection } from "@/lib/types";
import { DetectionClass } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DetectionTimelineProps {
  detections: Detection[];
  totalDuration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

const classColors: Partial<Record<DetectionClass, string>> = {
  [DetectionClass.PERSON]: "bg-cyan-400",
  [DetectionClass.CAR]: "bg-blue-400",
  [DetectionClass.TRUCK]: "bg-indigo-400",
  [DetectionClass.BICYCLE]: "bg-emerald-400",
  [DetectionClass.DOG]: "bg-amber-400",
  [DetectionClass.BACKPACK]: "bg-rose-400",
};

export function DetectionTimeline({
  detections,
  totalDuration,
  currentTime,
  onSeek,
}: DetectionTimelineProps) {
  const markers = useMemo(() => {
    const seen = new Set<string>();
    return detections
      .filter((d) => {
        // Deduplicate by track_id + class to avoid overlapping markers
        const key = `${d.track_id}-${d.class_name}-${Math.floor(d.timestamp_sec)}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .map((d) => ({
        ...d,
        position: (d.timestamp_sec / totalDuration) * 100,
      }));
  }, [detections, totalDuration]);

  const playheadPosition = (currentTime / totalDuration) * 100;

  return (
    <div className="relative">
      {/* Timeline bar */}
      <div
        className="relative h-8 cursor-pointer rounded-md bg-muted/50 overflow-hidden"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const pct = x / rect.width;
          onSeek(pct * totalDuration);
        }}
      >
        {/* Detection markers */}
        {markers.map((m) => (
          <div
            key={m.detection_id}
            className={cn(
              "absolute top-1 h-6 w-1 rounded-full opacity-70",
              classColors[m.class_name] || "bg-gray-400"
            )}
            style={{ left: `${m.position}%` }}
            title={`${m.class_name} at ${m.timestamp_sec.toFixed(1)}s (${(m.confidence * 100).toFixed(0)}%)`}
          />
        ))}

        {/* Playhead */}
        <div
          className="absolute top-0 h-full w-0.5 bg-white/80 z-10"
          style={{ left: `${playheadPosition}%` }}
        />
      </div>

      {/* Time labels */}
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground/60">
        <span>0:00</span>
        <span>
          {Math.floor(totalDuration / 60)}:{String(Math.floor(totalDuration % 60)).padStart(2, "0")}
        </span>
      </div>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-3">
        {Object.entries(classColors).map(([cls, color]) => (
          <div key={cls} className="flex items-center gap-1">
            <div className={cn("h-2 w-2 rounded-full", color)} />
            <span className="text-[10px] text-muted-foreground">{cls}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
