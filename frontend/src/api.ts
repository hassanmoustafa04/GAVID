import axios from "axios";

export interface ClassificationCandidate {
  label: string;
  confidence: number;
}

export interface InferenceResponse {
  top1: ClassificationCandidate;
  top5: ClassificationCandidate[];
  latency_ms: number;
  throughput_fps: number;
  engine: string;
  batch_size: number;
}

export interface GPUSample {
  gpu_util: number;
  mem_util: number;
  memory_used_mb: number;
  memory_total_mb: number;
  power_w: number;
  power_limit_w: number;
  temperature_c: number;
  timestamp: number;
}

export interface GPUResponse {
  samples: GPUSample[];
  source: string;
  interval_s: number;
  device?: string;
}

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? ""
});

export async function inferImage(file: File): Promise<InferenceResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await client.post<InferenceResponse>("/infer", formData, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return response.data;
}

export async function fetchGpuMetrics(): Promise<GPUResponse> {
  const response = await client.get<GPUResponse>("/metrics/gpu");
  return response.data;
}
