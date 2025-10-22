#!/bin/bash
set -e

echo "======================================"
echo "GAVID RunPod Setup Script"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on GPU instance
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Make sure you're on a GPU instance.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GPU detected:${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Navigate to workspace
WORKSPACE_DIR="/workspace/gavid"
mkdir -p /workspace
cd /workspace

# Clone repository if not exists
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo -e "${YELLOW}Cloning GAVID repository...${NC}"
    git clone https://github.com/hassanmoustafa04/GAVID.git gavid
    cd gavid
else
    echo -e "${YELLOW}Repository exists, pulling latest changes...${NC}"
    cd gavid
    git pull
fi

echo -e "${GREEN}✓ Repository ready${NC}"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Docker detected. Would you like to use Docker deployment? (y/n)${NC}"
    read -r USE_DOCKER

    if [ "$USE_DOCKER" = "y" ] || [ "$USE_DOCKER" = "Y" ]; then
        echo -e "${YELLOW}Building and starting Docker containers...${NC}"
        docker compose -f docker/docker-compose.yml up --build -d

        echo ""
        echo -e "${GREEN}======================================"
        echo "✓ GAVID is running in Docker!"
        echo -e "======================================${NC}"
        echo ""
        echo "Backend API: http://localhost:8000"
        echo "Frontend Dashboard: http://localhost:8080"
        echo "Health Check: http://localhost:8000/health"
        echo ""
        echo "To view logs: docker compose -f docker/docker-compose.yml logs -f"
        echo "To stop: docker compose -f docker/docker-compose.yml down"
        exit 0
    fi
fi

# Python setup
echo -e "${YELLOW}Setting up Python backend...${NC}"
cd backend

# Create virtual environment
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Create artifacts directory
mkdir -p artifacts

# Pre-compile TensorRT engine
echo -e "${YELLOW}Pre-compiling TensorRT engine (this may take 2-3 minutes)...${NC}"
python -m app.compile_engine --fp16 --max-batch 8

echo -e "${GREEN}✓ TensorRT engine compiled${NC}"
echo ""

# Set environment variables
export GAVID_DEVICE=cuda:0
export GAVID_MODEL_FP16=true
export GAVID_MODEL_ENGINE_PATH=/workspace/gavid/backend/artifacts/resnet18_fp16_engine.tsrt

# Check if Node.js is installed for frontend
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Installing Node.js...${NC}"
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs -qq
fi

# Setup frontend
echo -e "${YELLOW}Setting up frontend...${NC}"
cd ../frontend
npm install --silent
npm run build

echo -e "${GREEN}✓ Frontend built${NC}"
echo ""

# Install serve if not present
if ! command -v serve &> /dev/null; then
    npm install -g serve
fi

echo -e "${GREEN}======================================"
echo "✓ Setup Complete!"
echo -e "======================================${NC}"
echo ""
echo "To start GAVID:"
echo ""
echo "1. Start backend (in one terminal):"
echo "   cd /workspace/gavid/backend"
echo "   source .venv/bin/activate"
echo "   uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "2. Start frontend (in another terminal):"
echo "   cd /workspace/gavid/frontend"
echo "   npx serve -s dist -p 8080 --host 0.0.0.0"
echo ""
echo "Or use the quick start script:"
echo "   bash /workspace/gavid/scripts/runpod_start.sh"
echo ""
