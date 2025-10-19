#!/bin/bash
# Script to run tests with coverage

set -e

echo "🧪 Running tests..."
echo ""

# Run pytest with coverage
uv run pytest tests/ -v --cov=match_predictor --cov-report=html --cov-report=term

echo ""
echo "✅ Tests complete!"
echo "📊 Coverage report available at: htmlcov/index.html"
