"use client";

import { useEffect, useRef } from "react";
import type { HeatmapData } from "@/lib/types";

interface HeatmapCanvasProps {
  data: HeatmapData;
  width: number;
  height: number;
  opacity?: number;
}

function valueToColor(value: number): [number, number, number] {
  // Blue -> Cyan -> Green -> Yellow -> Red
  const clamped = Math.max(0, Math.min(1, value));
  let r = 0, g = 0, b = 0;
  if (clamped < 0.25) {
    const t = clamped / 0.25;
    r = 0; g = Math.round(t * 255); b = 255;
  } else if (clamped < 0.5) {
    const t = (clamped - 0.25) / 0.25;
    r = 0; g = 255; b = Math.round((1 - t) * 255);
  } else if (clamped < 0.75) {
    const t = (clamped - 0.5) / 0.25;
    r = Math.round(t * 255); g = 255; b = 0;
  } else {
    const t = (clamped - 0.75) / 0.25;
    r = 255; g = Math.round((1 - t) * 255); b = 0;
  }
  return [r, g, b];
}

export function HeatmapCanvas({
  data,
  width,
  height,
  opacity = 0.6,
}: HeatmapCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = width;
    canvas.height = height;

    const cellW = width / data.width;
    const cellH = height / data.height;

    ctx.clearRect(0, 0, width, height);

    for (let r = 0; r < data.height; r++) {
      for (let c = 0; c < data.width; c++) {
        const val = data.grid[r]?.[c] ?? 0;
        if (val < 0.05) continue;
        const [red, green, blue] = valueToColor(val);
        ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${val * opacity})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }
  }, [data, width, height, opacity]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0"
      style={{ width, height }}
    />
  );
}
