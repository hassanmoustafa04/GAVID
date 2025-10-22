#!/bin/bash

echo "Stopping GAVID..."
echo ""

# Stop Docker containers if running
if docker ps | grep -q gavid; then
    echo "Stopping Docker containers..."
    cd /workspace/gavid
    docker compose -f docker/docker-compose.yml down
    echo "✓ Docker containers stopped"
    exit 0
fi

# Kill Python processes
if pkill -f "uvicorn app.main:app"; then
    echo "✓ Backend stopped"
else
    echo "Backend was not running"
fi

# Kill frontend processes
if pkill -f "serve -s dist"; then
    echo "✓ Frontend stopped"
else
    echo "Frontend was not running"
fi

echo ""
echo "GAVID stopped."
