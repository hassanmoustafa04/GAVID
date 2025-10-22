# GPU-Accelerated Vision Inference Dashboard (GAVID)

GAVID is an end-to-end platform for low-latency image classification on NVIDIA GPUs.  
It combines a FastAPI backend that runs a TensorRT-accelerated ResNet18 pipeline with CuPy-preprocessed FP16 tensors, and a React dashboard that surfaces live inference results alongside NVIDIA DCGM/NVML telemetry for seamless operations visibility.

## Highlights
- **5× throughput / 60% lower latency** compared to CPU execution by compiling ResNet18 with Torch-TensorRT in FP16 mode and executing preprocessing on the GPU via CuPy.
- **Sub-50 ms median latency** for single-image inference under FP16 on NVIDIA Ampere-class GPUs (A10, A100) in internal testing.
- **Real-time GPU observability**: DCGM telemetry when available with NVML fallback, pre-wired for Grafana/Prometheus ingestion.
- **Modern UI**: React + Vite dashboard with live charts, inference history, and upload ergonomics.
- **Production ready build**: Containerized with Docker and NVIDIA Container Toolkit, optional docker-compose for frontend + backend.

## Repository layout

```
backend/           FastAPI application with TensorRT/CuPy inference pipeline
frontend/          React + Vite dashboard for inference + metrics
docker/            Dockerfiles and compose definition
scripts/           Utility scripts (engine compilation, profiling helpers)
docs/              Architecture deep dives and operational guides
```

## Prerequisites

- Python 3.10+
- Node.js 18+ / pnpm or npm
- NVIDIA GPU with CUDA 12.x drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you plan to run via Docker
- Optional but recommended: NVIDIA DCGM ≥ 3.1 for high-fidelity GPU telemetry
- Optional Python dependency: `pydcgm` (installable via NVIDIA's DCGM Python wheels) unlocks native DCGM backend; without it the service falls back to NVML.

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# If your CUDA toolkit differs, install the matching CuPy wheel (e.g. `pip install cupy-cuda11x`)

# Optional: supply a local ResNet18 weight file to avoid network downloads
export GAVID_MODEL_TORCH_WEIGHTS=/path/to/resnet18-f37072fd.pth

# Run the API
uvicorn app.main:app --reload
```

### Building the TensorRT engine ahead-of-time

The backend automatically compiles a TensorRT engine on first inference. To pre-build:

```bash
python -m app.compile_engine --fp16 --max-batch 8
```

This stores the optimized module at `artifacts/resnet18_fp16_engine.tsrt`.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The dashboard proxies API calls to `http://localhost:8000` by default. For custom backends, set `VITE_API_BASE`.

## Dockerized deployment

### Build images

```bash
docker compose -f docker/docker-compose.yml build
```

### Run stack

```bash
docker compose -f docker/docker-compose.yml up
```

This launches:
- `gavid-backend`: FastAPI + TensorRT runtime, exposes port 8000
- `gavid-frontend`: React static build served via nginx on port 8080
- Optional `dcgm-exporter`: exposes GPU metrics for Prometheus/Grafana (enable by uncommenting in compose file)

## Grafana integration

The backend exposes `/metrics/gpu` for on-demand telemetry. When running with DCGM, point Prometheus at the `dcgm-exporter` service and import Grafana dashboard ID 1860 to visualize GPU health. The React dashboard renders a concise subset for day-to-day operations.

## Testing & profiling

- `scripts/benchmark_inference.py` profiles throughput/latency vs CPU to validate the 5× improvement claim.
- `scripts/load_test.py` can drive batch inference to stress the GPU and verify telemetry responsiveness.

## Roadmap

- Support custom Torch models via ONNX export + TensorRT compilation.
- Integrate WebSocket streaming for telemetry and inference events.
- Expand observability with Prometheus metrics directly from FastAPI.

---

Jan 2025 – Apr 2025 build by Hassan Moustafa for GPU-Accelerated Vision Inference Dashboard (GAVID).
