#!/bin/bash
# Script to run the Streamlit app

set -e

echo "🎾 Starting Tennis Match Predictor Streamlit App..."
echo "🌐 App will be available at: http://localhost:8501"
echo ""

uv run streamlit run src/match_predictor/app/main.py
