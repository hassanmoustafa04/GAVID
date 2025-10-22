# GAVID Architecture

## Overview

GAVID splits responsibilities between a GPU-centric FastAPI backend and a React dashboard:

- **Backend** (`backend/app`): Hosts inference services, compiles ResNet18 to TensorRT, manages GPU telemetry, and exposes REST endpoints.
- **Frontend** (`frontend/src`): Presents inference controls, renders metrics charts, and surfaces operational alerts.
- **Observability**: NVIDIA DCGM (or NVML fallback) feeds GPU stats; optional dcgm-exporter + Grafana covers fleet-level monitoring.

## Backend pipeline

```
Upload image -> FastAPI /infer -> CuPy preprocessing (GPU) -> TensorRT FP16 engine
             -> Softmax + Top-k -> JSON response with latency + throughput telemetry
```

| Stage                | Implementation details                                                      |
| -------------------- | ---------------------------------------------------------------------------- |
| Input handling       | `UploadFile` → Pillow validation → CuPy transfer via DLPack                  |
| Preprocessing        | Resize on CPU (Pillow) → CuPy normalization → GPU tensor (FP32)              |
| TensorRT compilation | Torch-TensorRT dynamic profiles (min=1, opt=4, max=8) with FP16 precision    |
| Execution            | GPU path prioritizes TensorRT; fallback to PyTorch eager if unavailable      |
| Post-processing      | Softmax, top-5 extraction, throughput/latency metrics per request            |

## GPU telemetry

`metrics.py` exposes a pluggable collector:

- **Primary**: `pydcgm` (if installed) for DCGM field IDs 100/101/203/204/150/155/232.
- **Fallback**: `pynvml` covering utilization, memory, power, and temperature.
- **Stub**: Deterministic mock values for developer laptops without NVIDIA drivers.

Front-end polling cadence matches `Settings.metrics_update_interval_s` (defaults to 2s).

## Frontend data flow

1. `App.tsx` launches polling loop on `/metrics/gpu`.
2. Chart data stored in ring buffer (max 120 points).
3. Upload control sends `multipart/form-data` to `/infer`.
4. Toasts provide quick UX feedback; metrics cards render latest sample.

## Deployment

- **Docker Compose** orchestrates backend (GPU-enabled) + frontend + optional DCGM exporter.
- **Volumes** persist optimized TensorRT engine in `backend-artifacts`.
- In Kubernetes, translate Compose GPU reservations to device plugin requests.

## Latency goals

The pipeline targets <50 ms latency on Ampere-class hardware:

- CuPy GPU preprocessing removes CPU bottlenecks (~4 ms saving).
- TensorRT FP16 reduces compute time vs FP32 PyTorch (~5× throughput uplift).
- Optional `scripts/benchmark_inference.py` validates gains vs CPU fallback.

## Extensibility

- Swap ResNet18 with custom Torch modules by injecting a compatible `ResNet18Engine`.
- Extend metrics by subclassing `_MetricBackend` and wiring into `MetricsCollector`.
- Promote telemetry to Prometheus by wrapping collector outputs with `prometheus_client`.
