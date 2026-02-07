// VaultVision API Client
// Connects to backend at http://localhost:8001

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const DEFAULT_API_KEY = "vv_QiQHzLsnXAWlRXR3hEo59cpC2oWIFoNjv8ULH4jl_dI";

import type {
  VideoSummary,
  VideoStatusResponse,
  VideoUploadResponse,
  Detection,
  Event,
  HeatmapData,
  NLQueryRequest,
  NLQueryResponse,
  AnalyticsSummary,
} from "./types";

class VaultVisionAPI {
  private baseUrl: string;
  private apiKey: string | null = null;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
    // Load key from localStorage if available, otherwise use default
    if (typeof window !== "undefined") {
      this.apiKey = localStorage.getItem("vv_api_key") || DEFAULT_API_KEY;
    } else {
      this.apiKey = DEFAULT_API_KEY;
    }
  }

  setApiKey(key: string) {
    this.apiKey = key;
    if (typeof window !== "undefined") {
      localStorage.setItem("vv_api_key", key);
    }
  }

  getApiKey(): string | null {
    return this.apiKey;
  }

  getBaseUrl(): string {
    return this.baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const headers: Record<string, string> = {
      ...(options?.headers as Record<string, string>),
    };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    if (!(options?.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    const res = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    if (!res.ok) {
      throw new Error(`API error ${res.status}: ${await res.text()}`);
    }
    return res.json();
  }

  // Videos
  async uploadVideo(file: File): Promise<VideoUploadResponse> {
    const form = new FormData();
    form.append("file", file);
    return this.request("/api/v1/videos/upload", {
      method: "POST",
      body: form,
    });
  }

  async listVideos(): Promise<VideoSummary[]> {
    return this.request("/api/v1/videos");
  }

  async getVideoStatus(videoId: string): Promise<VideoStatusResponse> {
    return this.request(`/api/v1/videos/${videoId}/status`);
  }

  async getVideoDetections(videoId: string): Promise<Detection[]> {
    return this.request(`/api/v1/videos/${videoId}/detections`);
  }

  async getVideoEvents(videoId: string): Promise<Event[]> {
    return this.request(`/api/v1/videos/${videoId}/events`);
  }

  async getVideoHeatmap(videoId: string): Promise<HeatmapData> {
    return this.request(`/api/v1/videos/${videoId}/heatmap`);
  }

  async deleteVideo(videoId: string): Promise<void> {
    await this.request(`/api/v1/videos/${videoId}`, { method: "DELETE" });
  }

  // Video streaming URL
  getVideoStreamUrl(videoId: string): string {
    return `${this.baseUrl}/api/v1/videos/${videoId}/stream`;
  }

  // Query
  async queryNL(req: NLQueryRequest): Promise<NLQueryResponse> {
    return this.request("/api/v1/query", {
      method: "POST",
      body: JSON.stringify(req),
    });
  }

  // Analytics
  async getAnalyticsSummary(): Promise<AnalyticsSummary> {
    return this.request("/api/v1/analytics/summary");
  }

  // API Keys
  async listApiKeys(): Promise<Array<{ id: string; key: string; name: string; created_at: string }>> {
    return this.request("/api/v1/keys");
  }

  async createApiKey(name: string): Promise<{ id: string; key: string; name: string; created_at: string }> {
    return this.request("/api/v1/keys", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
  }

  async deleteApiKey(keyId: string): Promise<void> {
    await this.request(`/api/v1/keys/${keyId}`, { method: "DELETE" });
  }

  // WebSocket for processing progress
  connectWebSocket(videoId: string, onMessage: (msg: unknown) => void): WebSocket {
    const wsUrl = this.baseUrl.replace(/^http/, "ws") + `/ws/progress/${videoId}`;
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      onMessage(JSON.parse(event.data));
    };
    return ws;
  }
}

export const api = new VaultVisionAPI(API_BASE);
