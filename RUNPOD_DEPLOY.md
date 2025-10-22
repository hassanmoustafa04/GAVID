# GAVID RunPod Deployment Guide

Quick guide to deploy GAVID on your RunPod GPU instance.

## Prerequisites

- Active RunPod GPU pod (any GPU: A4000, A5000, A100, RTX 4090, etc.)
- SSH access or web terminal access to your pod

## Quick Start (One Command)

Connect to your RunPod instance and run:

```bash
curl -sSL https://raw.githubusercontent.com/hassanmoustafa04/GAVID/main/scripts/runpod_setup.sh | bash
```

Then start GAVID:

```bash
bash /workspace/gavid/scripts/runpod_start.sh
```

## Manual Setup

### Step 1: Connect to RunPod

SSH into your instance:
```bash
ssh root@<your-runpod-ip> -p <port>
```

Or use the "Connect" button in RunPod's web interface.

### Step 2: Clone Repository

```bash
cd /workspace
git clone https://github.com/hassanmoustafa04/GAVID.git gavid
cd gavid
```

### Step 3: Run Setup Script

```bash
chmod +x scripts/*.sh
bash scripts/runpod_setup.sh
```

This will:
- Verify GPU availability
- Install all dependencies
- Pre-compile TensorRT engine (FP16 optimized)
- Build frontend

### Step 4: Start GAVID

```bash
bash scripts/runpod_start.sh
```

## Accessing GAVID

### Local Access (from RunPod terminal)
- Backend API: http://localhost:8000
- Frontend Dashboard: http://localhost:8080

### External Access (from your browser)

RunPod provides port forwarding. You have two options:

#### Option 1: RunPod's Built-in Proxy
1. Go to your pod page on RunPod
2. Click "Connect" → "HTTP Service"
3. Add ports: **8000** (backend) and **8080** (frontend)
4. Use the provided URLs

#### Option 2: SSH Tunnel (More reliable)
From your local machine:

```bash
# Forward frontend
ssh -L 8080:localhost:8080 root@<runpod-ip> -p <port>

# In another terminal, forward backend
ssh -L 8000:localhost:8000 root@<runpod-ip> -p <port>
```

Then open http://localhost:8080 in your browser.

## Docker Deployment (Alternative)

If you prefer Docker:

```bash
cd /workspace/gavid
docker compose -f docker/docker-compose.yml up --build -d
```

Access:
- Frontend: http://localhost:8080
- Backend: http://localhost:8000

View logs:
```bash
docker compose -f docker/docker-compose.yml logs -f
```

Stop:
```bash
docker compose -f docker/docker-compose.yml down
```

## Testing Your Deployment

### 1. Check GPU Detection
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "device": "cuda:0"
}
```

### 2. Test GPU Metrics
```bash
curl http://localhost:8000/metrics/gpu
```

Should return GPU utilization, memory, power, temperature.

### 3. Test Inference

Download a test image:
```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg -O /tmp/test.jpg
```

Run inference:
```bash
curl -X POST http://localhost:8000/infer \
  -F "file=@/tmp/test.jpg"
```

Expected response:
```json
{
  "top1_class": "golden_retriever",
  "top1_confidence": 0.89,
  "top5": [...],
  "latency_ms": 28.5,
  "throughput_fps": 35.1,
  "engine": "TensorRT"
}
```

## Performance Expectations

Based on your GPU type:

| GPU | Expected Latency | Throughput |
|-----|-----------------|------------|
| RTX 4090 | 15-25ms | 40-60 FPS |
| A4000 | 25-35ms | 28-40 FPS |
| A5000 | 20-30ms | 33-50 FPS |
| A100 | 15-20ms | 50-65 FPS |

## Managing GAVID

### View Logs
```bash
# Backend logs
tail -f /workspace/gavid_backend.log

# Frontend logs
tail -f /workspace/gavid_frontend.log
```

### Stop GAVID
```bash
bash /workspace/gavid/scripts/runpod_stop.sh
```

### Restart GAVID
```bash
bash /workspace/gavid/scripts/runpod_stop.sh
bash /workspace/gavid/scripts/runpod_start.sh
```

### Update Code
```bash
cd /workspace/gavid
git pull
bash scripts/runpod_stop.sh
bash scripts/runpod_setup.sh
bash scripts/runpod_start.sh
```

## Persistence

RunPod's `/workspace` directory is persistent across pod stops/starts.

To preserve your TensorRT engine:
```bash
# Engine is already in /workspace/gavid/backend/artifacts/
# It will persist automatically
```

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi
```
If this fails, your pod doesn't have GPU access. Recreate with GPU enabled.

### Port Already in Use
```bash
bash /workspace/gavid/scripts/runpod_stop.sh
```
Then start again.

### TensorRT Compilation Failed
Check GPU compatibility:
```bash
python3 -c "import torch; print(torch.cuda.get_device_capability())"
```

For older GPUs, disable TensorRT and use PyTorch:
```bash
export GAVID_MODEL_ENGINE_PATH=""
```

### DCGM Metrics Not Available
Install DCGM (optional):
```bash
apt-get update
apt-get install -y datacenter-gpu-manager
```

Or use NVML backend (automatic fallback).

## Cost Optimization

RunPod charges by the hour. To minimize costs:

1. **Use Spot Instances**: 50-80% cheaper than on-demand
2. **Stop pod when not in use**: Billing stops when pod is stopped
3. **Use smaller GPUs**: RTX 4090 sufficient for most workloads

## Benchmarking

Run performance tests:

```bash
cd /workspace/gavid/backend
source .venv/bin/activate
python ../scripts/benchmark_inference.py
```

Load test:
```bash
python ../scripts/load_test.py --concurrent 50
```

## Support

- Repository: https://github.com/hassanmoustafa04/GAVID
- Issues: https://github.com/hassanmoustafa04/GAVID/issues

---

**Built by Hassan Moustafa | Jan 2025 – Apr 2025**
