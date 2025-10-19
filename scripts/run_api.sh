#!/bin/bash
# Script to run the FastAPI service

set -e

echo "🚀 Starting Tennis Match Predictor API..."
echo "📡 API will be available at: http://localhost:8000"
echo "📚 API docs will be available at: http://localhost:8000/docs"
echo ""

# Check if running in development or production mode
MODE=${1:-dev}

if [ "$MODE" = "dev" ]; then
    echo "Running in development mode with auto-reload..."
    uv run uvicorn match_predictor.api.main:app --reload --host 0.0.0.0 --port 8000
else
    echo "Running in production mode..."
    uv run uvicorn match_predictor.api.main:app --host 0.0.0.0 --port 8000 --workers 4
fi
