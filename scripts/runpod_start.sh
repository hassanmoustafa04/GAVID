#!/bin/bash
set -e

echo "Starting GAVID on RunPod..."
echo ""

WORKSPACE_DIR="/workspace/gavid"

if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "ERROR: GAVID not found. Run runpod_setup.sh first."
    exit 1
fi

cd "$WORKSPACE_DIR"

# Check if Docker containers are running
if docker ps | grep -q gavid-backend; then
    echo "GAVID is already running in Docker."
    echo ""
    echo "Backend API: http://localhost:8000"
    echo "Frontend Dashboard: http://localhost:8080"
    echo ""
    echo "To view logs: docker compose -f docker/docker-compose.yml logs -f"
    exit 0
fi

# Start backend in background
echo "Starting backend..."
cd backend
source .venv/bin/activate

# Export environment variables
export GAVID_DEVICE=cuda:0
export GAVID_MODEL_FP16=true
export GAVID_MODEL_ENGINE_PATH=/workspace/gavid/backend/artifacts/resnet18_fp16_engine.tsrt

# Kill any existing uvicorn processes
pkill -f "uvicorn app.main:app" || true

# Start backend
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > /workspace/gavid_backend.log 2>&1 &
BACKEND_PID=$!

echo "Backend started (PID: $BACKEND_PID)"
sleep 3

# Start frontend
echo "Starting frontend..."
cd ../frontend

# Kill any existing serve processes
pkill -f "serve -s dist" || true

# Start frontend
nohup npx serve -s dist -p 8080 --host 0.0.0.0 > /workspace/gavid_frontend.log 2>&1 &
FRONTEND_PID=$!

echo "Frontend started (PID: $FRONTEND_PID)"
sleep 2

echo ""
echo "======================================"
echo "✓ GAVID is running!"
echo "======================================"
echo ""
echo "Backend API: http://localhost:8000"
echo "Frontend Dashboard: http://localhost:8080"
echo ""
echo "Backend logs: tail -f /workspace/gavid_backend.log"
echo "Frontend logs: tail -f /workspace/gavid_frontend.log"
echo ""
echo "To stop GAVID:"
echo "  bash /workspace/gavid/scripts/runpod_stop.sh"
echo ""

# Test backend health
echo "Testing backend health..."
sleep 2
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is healthy"
else
    echo "⚠ Backend health check failed. Check logs."
fi

echo ""
