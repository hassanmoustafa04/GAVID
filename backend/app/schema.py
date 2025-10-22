from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ClassificationCandidate(BaseModel):
    label: str = Field(..., description="Human readable label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability score.")


class InferenceResponse(BaseModel):
    top1: ClassificationCandidate
    top5: List[ClassificationCandidate]
    latency_ms: float = Field(..., ge=0.0)
    throughput_fps: float = Field(..., ge=0.0)
    engine: str = Field(..., description="Name of the execution backend used.")
    batch_size: int = Field(..., gt=0)


class GPUUtilizationSample(BaseModel):
    gpu_util: float = Field(..., description="GPU utilization percentage.")
    mem_util: float = Field(..., description="Copy engine utilization percentage.")
    memory_used_mb: float = Field(..., description="Framebuffer used in MiB.")
    memory_total_mb: float = Field(..., description="Framebuffer total in MiB.")
    power_w: float = Field(..., description="Instantaneous power draw in watts.")
    power_limit_w: float = Field(..., description="Power cap in watts.")
    temperature_c: float = Field(..., description="GPU temperature in Celsius.")
    timestamp: float = Field(..., description="Epoch timestamp for the sample.")


class GPUUtilizationResponse(BaseModel):
    samples: List[GPUUtilizationSample]
    source: str = Field(..., description="Telemetry provider (dcgm or fallback).")
    interval_s: float = Field(..., description="Sampling interval in seconds.")
    device: Optional[str] = Field(None, description="Device identifier.")
