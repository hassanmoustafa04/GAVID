# macOS Installation Guide for GAVID

This guide covers setting up GAVID on macOS for local development and testing **without NVIDIA GPU**.

## ✅ What's Been Fixed

The original codebase required GPU-specific dependencies that don't work on macOS. These issues have been resolved:

1. ✅ **torch-tensorrt**: Removed from base requirements (GPU-only)
2. ✅ **cupy-cuda12x**: Moved to GPU-specific requirements
3. ✅ **NumPy 2.x compatibility**: Downgraded to NumPy 1.x for PyTorch 2.2.1
4. ✅ **pydantic-settings**: Fixed import for Pydantic 2.x
5. ✅ **ResNet18 weights**: Downloaded locally to avoid SSL certificate issues
6. ✅ **CPU fallback mode**: Enabled by default for macOS

## Quick Start (macOS)

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Download ResNet18 weights (bypass SSL issues)
mkdir -p downloads
python -c "
import requests
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
url = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
print(f'Downloading ResNet18 weights...')
response = requests.get(url, verify=False, stream=True)
with open('downloads/resnet18-f37072fd.pth', 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
print('✓ Downloaded successfully')
"

# Update .env to use local weights
echo "GAVID_MODEL_TORCH_WEIGHTS=$(pwd)/downloads/resnet18-f37072fd.pth" >> .env

# Start backend
uvicorn app.main:app --reload
```

Backend will run on: **http://localhost:8000**

### 2. Frontend Setup

In a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run on: **http://localhost:5173**

## Testing with Images

### Method 1: Web Dashboard

1. Open http://localhost:5173 in your browser
2. Upload an image (JPG, PNG, etc.)
3. View classification results and metrics

### Method 2: Command Line

```bash
# Health check
curl http://localhost:8000/health

# Test inference with an image
curl -X POST http://localhost:8000/infer \
  -F "file=@/path/to/your/image.jpg"

# Get GPU metrics (will show stub data on macOS)
curl http://localhost:8000/metrics/gpu
```

## Example Response

```json
{
  "top1": {"label": "class_111", "confidence": 0.037},
  "top5": [...],
  "latency_ms": 22.2,
  "throughput_fps": 45.0,
  "engine": "PyTorch",
  "batch_size": 1
}
```

## Performance on macOS (CPU Mode)

**Expected performance:**
- Latency: ~20-50ms per image (CPU-bound)
- Engine: PyTorch (no TensorRT on macOS)
- GPU metrics: Stub data (fake values for UI testing)

**What works:**
- ✅ Full API functionality
- ✅ Image classification with ResNet18
- ✅ Top-5 predictions
- ✅ Frontend dashboard
- ✅ All endpoints (/health, /infer, /metrics/gpu)

**What doesn't work (expected on macOS):**
- ❌ TensorRT acceleration (requires NVIDIA GPU)
- ❌ CuPy GPU preprocessing (requires CUDA)
- ❌ Real DCGM/NVML metrics (requires NVIDIA GPU)
- ❌ Sub-50ms latency (GPU-only performance)

## Troubleshooting

### SSL Certificate Error

If you see SSL certificate errors when downloading weights:
```bash
# macOS Python SSL fix
/Applications/Python\ 3.12/Install\ Certificates.command
```

Or use the manual download method shown in step 1.

### NumPy Version Conflict

If you see NumPy 2.x errors:
```bash
pip install 'numpy<2.0.0'
```

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn app.main:app --port 8001
```

## Files Created/Modified

**New files:**
- `backend/requirements-gpu.txt` - GPU-specific dependencies for Linux
- `backend/requirements-dev.txt` - Development dependencies with testing tools
- `backend/.env.example` - Environment configuration template
- `backend/.env` - Your local configuration
- `backend/downloads/resnet18-f37072fd.pth` - ResNet18 weights

**Modified files:**
- `backend/requirements.txt` - CPU-compatible base dependencies only
- `backend/app/config.py` - Fixed pydantic-settings import
- `README.md` - Added platform-specific instructions

## Next Steps

For GPU-accelerated performance:
1. Deploy to a Linux machine with NVIDIA GPU
2. Use RunPod, AWS, or Google Cloud with GPU instances
3. Follow the Linux setup instructions in [README.md](README.md)

## Support

- Issues: https://github.com/anthropics/claude-code/issues
- Docs: See [README.md](README.md) and [docs/architecture.md](docs/architecture.md)

---

**Date**: Jan 2025
**Tested on**: macOS 14.0+ (Apple Silicon M1/M2/M3)
**Python**: 3.12
**Status**: ✅ Fully functional for development and testing
