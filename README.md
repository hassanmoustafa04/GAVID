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

### For GPU-Accelerated Production (Linux + NVIDIA GPU)
- Python 3.10+
- Node.js 18+ / pnpm or npm
- NVIDIA GPU with CUDA 12.x drivers
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) if you plan to run via Docker
- Optional but recommended: NVIDIA DCGM ≥ 3.1 for high-fidelity GPU telemetry
- Optional Python dependency: `pydcgm` (installable via NVIDIA's DCGM Python wheels) unlocks native DCGM backend; without it the service falls back to NVML.

### For CPU-Only Development (macOS / Windows)
- Python 3.10+
- Node.js 18+ / pnpm or npm
- No GPU required - uses CPU fallback mode with stub metrics

## Backend setup

### Option 1: macOS / CPU-Only Development

Perfect for local development and testing without NVIDIA GPU:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt

# Copy environment template and configure for CPU mode
cp .env.example .env
# Edit .env and ensure: GAVID_DEVICE=cpu and GAVID_ALLOW_CPU_FALLBACK=true

# Run the API
uvicorn app.main:app --reload
```

**What works in CPU mode:**
- ✅ Full API functionality (all endpoints)
- ✅ PyTorch CPU inference (slower than GPU, no TensorRT)
- ✅ Image classification with ResNet18
- ✅ Stub GPU metrics (fake data for UI testing)
- ✅ Frontend development
- ❌ TensorRT acceleration (requires NVIDIA GPU)
- ❌ CuPy GPU preprocessing (requires CUDA)
- ❌ Real DCGM/NVML metrics (requires NVIDIA GPU)

### Option 2: Linux with NVIDIA GPU (Production)

For full GPU acceleration with TensorRT:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install base dependencies
pip install -r requirements.txt

# Install GPU-specific dependencies
pip install -r requirements-gpu.txt

# Optional: supply a local ResNet18 weight file to avoid network downloads
export GAVID_MODEL_TORCH_WEIGHTS=/path/to/resnet18-f37072fd.pth

# Copy and configure environment
cp .env.example .env
# Edit .env and ensure: GAVID_DEVICE=cuda:0 and GAVID_MODEL_FP16=true

# Run the API
uvicorn app.main:app --reload
```

### Building the TensorRT engine ahead-of-time (GPU only)

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

## Testing with Images

### Method 1: Web Dashboard (Recommended)

1. Start backend and frontend (see setup instructions above)
2. Open browser:
   - Native setup: `http://localhost:5173`
   - Docker setup: `http://localhost:8080`
3. Upload an image via the UI
4. View results:
   - Top-5 classification predictions with confidence scores
   - Latency and throughput metrics
   - Live GPU/CPU metrics charts

### Method 2: API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Upload an image for inference
curl -X POST http://localhost:8000/infer \
  -F "file=@/path/to/your/image.jpg"

# Get GPU metrics (5 samples)
curl http://localhost:8000/metrics/gpu?samples=5
```

**Example response:**
```json
{
  "top1": {"label": "golden retriever", "confidence": 0.89},
  "top5": [...],
  "latency_ms": 28.5,
  "throughput_fps": 35.1,
  "engine": "TensorRT",
  "batch_size": 1
}
```

### Method 3: Automated Testing

```bash
# Benchmark CPU vs GPU performance (GPU only)
python scripts/benchmark_inference.py --iters 25

# Load test with random images
python scripts/load_test.py --url http://localhost:8000/infer --requests 50
```

## Grafana integration

The backend exposes `/metrics/gpu` for on-demand telemetry. When running with DCGM, point Prometheus at the `dcgm-exporter` service and import Grafana dashboard ID 1860 to visualize GPU health. The React dashboard renders a concise subset for day-to-day operations.

## Performance benchmarking

- `scripts/benchmark_inference.py` profiles throughput/latency vs CPU to validate the 5× improvement claim (GPU setup only).
- `scripts/load_test.py` can drive batch inference to stress the system and verify telemetry responsiveness.

## Roadmap

- Support custom Torch models via ONNX export + TensorRT compilation.
- Integrate WebSocket streaming for telemetry and inference events.
- Expand observability with Prometheus metrics directly from FastAPI.

---

Jan 2025 – Apr 2025 build by Hassan Moustafa for GPU-Accelerated Vision Inference Dashboard (GAVID).
